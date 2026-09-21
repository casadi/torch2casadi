"""Fixed-shape tensor models, with dynamic derivative direction counts."""
from copy import deepcopy
from pathlib import Path
import tempfile
import re

import onnx
import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_map


def export(model, example_input, directory, *, name="f", forward=False, hessian=True,
           input_names=None, is_diff_in=None, overwrite=False):
    """Export evaluation, adjoint and forward-over-adjoint ONNX models.

    example_input is a tensor or a tuple of positional tensor arguments.
    is_diff_in selects whole arguments to differentiate (default: all).
    Other arguments remain runtime inputs, including in derivative graphs.
    hessian=False skips the forward-over-adjoint graph, whose cost grows with the
    number of forward directions (e.g. weights exposed as inputs for training).
    Inputs/output use PyTorch flatten order, exposed as CasADi column vectors.
    ONNX structure is checked before files are published.
    overwrite replaces an existing family only after all graphs pass that check.
    """
    if not name.isidentifier():
        raise ValueError("name must be a Python identifier")
    samples = example_input if isinstance(example_input, tuple) else (example_input,)
    if not samples or any(not isinstance(v, torch.Tensor) or
                         v.dtype not in (torch.float32, torch.float64) for v in samples):
        raise TypeError("example_input must be a floating tensor or a nonempty tuple of them")
    dtype = samples[0].dtype
    if any(v.dtype != dtype for v in samples):
        raise TypeError("all example inputs must have the same dtype")
    count = len(samples)
    if input_names is None:
        input_names = ["x"] if count == 1 else [f"x{i}" for i in range(count)]
    input_names = list(input_names)
    if len(input_names) != count or len(set(input_names)) != count or any(
            not isinstance(n, str) or not n.isidentifier() or n == "y" or
            n.startswith(("adj_", "fwd_", "out_", "jac_")) for n in input_names):
        raise ValueError("input_names must be unique identifiers, one per input, without reserved names")
    if is_diff_in is None:
        is_diff_in = [True] * count
    is_diff_in = list(is_diff_in)
    if len(is_diff_in) != count or any(not isinstance(v, bool) for v in is_diff_in):
        raise ValueError("is_diff_in must contain one boolean per input")
    active = tuple(i for i, enabled in enumerate(is_diff_in) if enabled)
    directory = Path(directory)
    family = re.compile(r"(?:(?:fwd|adj|jac)_)*"+re.escape(name)+r"\.onnx")
    existing = {p for p in directory.glob("*.onnx") if family.fullmatch(p.name)}
    if existing and not overwrite:
        raise FileExistsError("Use a fresh directory or overwrite=True to replace this model family")
    model = deepcopy(model).cpu().eval().requires_grad_(False)
    samples = tuple(v.detach().cpu().contiguous() for v in samples)
    shapes = [tuple(v.shape) for v in samples]
    sizes = [v.numel() for v in samples]
    with torch.no_grad():
        example_output = model(*samples)
    if not isinstance(example_output, torch.Tensor) or example_output.dtype != dtype:
        raise TypeError("model must return one tensor with the input dtype")
    ny = example_output.numel()
    if not all(sizes) or not ny:
        raise ValueError("empty model inputs/outputs are not supported")

    def primal(*flat):
        return model(*(v.reshape(shape) for v, shape in zip(flat, shapes))).reshape(-1)

    def adjoint(flat, w):
        def weighted(*args):
            return (primal(*args[:-1])*args[-1]).sum()
        gradient = torch.func.grad(weighted, argnums=active)
        return torch.vmap(gradient, in_dims=(None,)*count+(0,))(*flat, w)

    def jvp(fn, args, directions):
        # Reverse-over-reverse avoids forward-AD operators unsupported by export.
        z = tree_map(torch.zeros_like, fn(*args))
        def pullback(z):
            return torch.func.vjp(fn, *args)[1](z)
        return torch.func.vjp(pullback, z)[1](directions)[0]

    def reduced(flat, with_adjoint=False):
        # Nondifferentiable inputs remain captured values, not AD arguments.
        def fn(*args):
            values = list(flat)
            for i, v in zip(active, args):
                values[i] = v
            return adjoint(tuple(values), args[-1]) if with_adjoint else primal(*values)
        return fn

    def f(*args):
        return primal(*(v[:, 0] for v in args))[:, None]

    def a(*args):
        flat = tuple(v[:, 0] for v in args[:count])
        return tuple(v.T for v in adjoint(flat, args[count].T))

    def mixed(args, jvp_impl):
        flat = tuple(v[:, 0] for v in args[:count])
        w = args[count].T
        directions = tuple(v.T for v in args[count+1:-1])
        nf = directions[0].shape[0]
        dw = args[-1].T.reshape(nf, w.shape[0], ny)
        fn = reduced(flat, with_adjoint=True)
        point = tuple(flat[i] for i in active)+(w,)
        result = torch.vmap(lambda *d: jvp_impl(fn, point, d))(*directions, dw)
        return tuple(v.reshape(-1, sizes[i]).T for i, v in zip(active, result))

    def fa(*args):
        if count == 1:
            return mixed(args, jvp)
        w = args[count].T
        directions = tuple(v.T for v in args[count+1:-1])
        nf = directions[0].shape[0]
        dw = args[-1].T.reshape(nf, w.shape[0], ny)
        result = torch.vmap(single_mixed, in_dims=(None,)*(count+1)+(0,)*(len(active)+1))(
            *args[:count], w, *directions, dw)
        return tuple(v.reshape(-1, sizes[i]).T for i, v in zip(active, result))

    def directional(args, jvp_impl):
        flat = tuple(v[:, 0] for v in args[:count])
        fn = reduced(flat)
        point = tuple(flat[i] for i in active)
        return torch.vmap(lambda *d: jvp_impl(fn, point, d))(
            *(v.T for v in args[count:])).T

    def fw(*args):
        if count == 1:
            return directional(args, jvp)
        return torch.vmap(single_forward, in_dims=(None,)*count+(0,)*len(active))(
            *args[:count], *(v.T for v in args[count:])).T

    generator = torch.Generator().manual_seed(2026)
    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=dtype)
    xs = tuple(v.reshape(-1, 1) for v in samples)
    # Distinct example sizes avoid accidental equality between symbolic/model axes.
    na, nf = 7, 11
    w, dw = rand(ny, na), rand(ny, na*nf)
    vs = tuple(rand(sizes[i], nf) for i in active)
    N, F, P = (torch.export.Dim(s) for s in ("nadj", "nfwd", "nsens"))
    static = ({},)*count
    adj_names = ["adj_"+input_names[i] for i in active]
    fwd_names = ["fwd_"+input_names[i] for i in active]
    specifications = [(name, f, xs, input_names, ["y"], static)]
    if active:
        specifications.append(
            ("adj_"+name, a, xs+(w,), input_names+["adj_y"], adj_names, static+({1:N},)))
        if hessian:
            specifications.append(
                ("fwd_adj_"+name, fa, xs+(w,)+vs+(dw,),
                 input_names+["adj_y"]+fwd_names+["fwd_adj_y"],
                 ["fwd_"+n for n in adj_names], static+({1:N},)+({1:F},)*len(active)+({1:P},)))
        if forward:
            specifications.append(("fwd_"+name, fw, xs+vs, input_names+fwd_names,
                                   ["fwd_y"], static+({1:F},)*len(active)))
    decompositions = get_decompositions([torch.ops.aten.tanh_backward.default])
    decompositions[torch.ops.aten._unsafe_view.default] = lambda t, shape: t.reshape(shape)
    def slice_backward(gradient, input_sizes, dim, start, end, step):
        # Padding avoids slice-backward's vmap rule specializing the outer seed count.
        if step == 1:
            dim = dim % gradient.ndim
            size = input_sizes[dim]
            start = max(0, min(size, start if start >= 0 else start+size))
            padding = [0]*(2*(gradient.ndim-dim-1))+[start, size-start-gradient.shape[dim]]
            return torch.nn.functional.pad(gradient, padding)
        shape = list(gradient.shape)
        shape[dim] = input_sizes[dim]
        return torch.slice_scatter(gradient.new_zeros(shape), gradient, dim, start, end, step)
    decompositions[torch.ops.aten.slice_backward.default] = slice_backward
    if active and count > 1:
        # Separate arguments introduce slice gradients when concatenated.
        # Trace AD before batching: slice-backward's vmap rule fixes the outer seed count.
        if hessian:
            def one_mixed(*args):
                flat = tuple(v[:, 0] for v in args[:count])
                fn = reduced(flat, with_adjoint=True)
                point = tuple(flat[i] for i in active)+(args[count],)
                return jvp(fn, point, args[count+1:])
            single_mixed = make_fx(one_mixed, decomposition_table=decompositions,
                tracing_mode="symbolic", _allow_non_fake_inputs=True)(
                    *xs, w.T, *(v[:, 0] for v in vs), dw[:, :na].T)
        if forward:
            def one_forward(*args):
                flat = tuple(v[:, 0] for v in args[:count])
                return jvp(reduced(flat), tuple(flat[i] for i in active), args[count:])
            single_forward = make_fx(one_forward, decomposition_table=decompositions,
                tracing_mode="symbolic", _allow_non_fake_inputs=True)(*xs, *(v[:, 0] for v in vs))
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        temporary = Path(temporary)
        for filename, fn, args, inputs, outputs, dynamic in specifications:
            graph = make_fx(fn, decomposition_table=decompositions, tracing_mode="symbolic",
                            _allow_non_fake_inputs=True)(*args)
            target = temporary / (filename+".onnx")
            torch.onnx.export(graph, args, str(target), dynamo=True, opset_version=18,
                              input_names=inputs, output_names=outputs, dynamic_shapes=dynamic,
                              external_data=False)
            onnx_model = onnx.load(target)
            symbols = {}
            for value, dims in zip(onnx_model.graph.input, dynamic):
                for axis, dim in dims.items():
                    symbol = value.type.tensor_type.shape.dim[axis].dim_param
                    if symbol:
                        symbols[symbol] = dim.__name__
            for value in list(onnx_model.graph.input) + list(onnx_model.graph.output) + list(onnx_model.graph.value_info):
                for dim in value.type.tensor_type.shape.dim:
                    if dim.dim_param in symbols:
                        dim.dim_param = symbols[dim.dim_param]
            onnx.checker.check_model(onnx_model)
            onnx.save(onnx_model, target)
        for filename, *_ in specifications:
            (temporary/(filename+".onnx")).replace(directory/(filename+".onnx"))
        written = {directory/(filename+".onnx") for filename, *_ in specifications}
        for stale in existing-written:
            stale.unlink(missing_ok=True)
    return directory/(name+".onnx")
