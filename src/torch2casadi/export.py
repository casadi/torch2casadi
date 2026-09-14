"""Fixed-shape tensor models, with dynamic derivative direction counts."""
from copy import deepcopy
from pathlib import Path
import tempfile

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


def export(model, example_input, directory, *, name="f", forward=False):
    """Export evaluation, adjoint and forward-over-adjoint ONNX models.

    The module accepts one floating tensor and returns one floating tensor.
    Model axes are fixed by example_input; derivative seed counts are dynamic.
    Tensor elements use PyTorch's flatten order, exposed as CasADi column vectors.
    Validation against PyTorch AD is mandatory before files are published.
    """
    if not name.isidentifier():
        raise ValueError("name must be a Python identifier")
    if not isinstance(example_input, torch.Tensor) or example_input.dtype not in (torch.float32, torch.float64):
        raise TypeError("example_input must be a float32 or float64 tensor")
    directory = Path(directory)
    if directory.exists() and any(p.name == name+".onnx" or p.name.endswith("_"+name+".onnx")
                                  for p in directory.glob("*.onnx")):
        raise FileExistsError("Use a fresh directory to avoid mixing derivative families")
    model = deepcopy(model).cpu().eval().requires_grad_(False)
    example_input = example_input.detach().cpu().contiguous()
    shape = tuple(example_input.shape)
    with torch.no_grad():
        example_output = model(example_input)
    if not isinstance(example_output, torch.Tensor) or example_output.dtype != example_input.dtype:
        raise TypeError("model must return one tensor with the input dtype")
    nx, ny = example_input.numel(), example_output.numel()
    if not nx or not ny:
        raise ValueError("empty model inputs/outputs are not supported")

    def primal(x):
        return model(x.reshape(shape)).reshape(-1)

    def adjoint(x, w):
        return torch.vmap(torch.func.grad(lambda x, w: (primal(x)*w).sum()),
                          in_dims=(None, 0))(x, w)

    def jvp(fn, args, directions):
        # Reverse-over-reverse avoids forward-AD operators unsupported by export.
        z = torch.zeros_like(fn(*args))
        def pullback(z):
            return torch.func.vjp(fn, *args)[1](z)
        return torch.func.vjp(pullback, z)[1](directions)[0]

    def fwd_adjoint(x, w, v, dw):
        return torch.vmap(lambda vi, dwi: jvp(adjoint, (x, w), (vi, dwi)))(v, dw)

    def f(x):
        return primal(x[:, 0])[:, None]

    def a(x, w):
        return adjoint(x[:, 0], w.T).T

    def fa(x, w, v, dw):
        result = fwd_adjoint(x[:, 0], w.T, v.T, dw.T.reshape(v.shape[1], w.shape[1], ny))
        return result.reshape(-1, nx).T

    def fw(x, v):
        return torch.vmap(lambda vi: jvp(primal, (x[:, 0],), (vi,)))(v.T).T

    generator = torch.Generator().manual_seed(2026)
    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=example_input.dtype)
    x = example_input.reshape(-1, 1)
    # Distinct example sizes avoid accidental equality between symbolic/model axes.
    na, nf = 7, 11
    w, v, dw = rand(ny, na), rand(nx, nf), rand(ny, na*nf)
    N, F, P = (torch.export.Dim(s) for s in ("nadj", "nfwd", "nsens"))
    specifications = [
        (name, f, (x,), ["x"], "y", ({},)),
        ("adj_"+name, a, (x,w), ["x","adj_y"], "adj_x", ({},{1:N})),
        ("fwd_adj_"+name, fa, (x,w,v,dw), ["x","adj_y","fwd_x","fwd_adj_y"],
         "fwd_adj_x", ({},{1:N},{1:F},{1:P})),
    ]
    if forward:
        specifications.append(("fwd_"+name,fw,(x,v),["x","fwd_x"],"fwd_y",({},{1:F})))
    decompositions = get_decompositions([torch.ops.aten.tanh_backward.default])
    decompositions[torch.ops.aten._unsafe_view.default] = lambda t, shape: t.reshape(shape)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=directory) as temporary:
        temporary = Path(temporary)
        for filename, fn, args, inputs, output, dynamic in specifications:
            graph = make_fx(fn, decomposition_table=decompositions, tracing_mode="symbolic",
                            _allow_non_fake_inputs=True)(*args)
            target = temporary / (filename+".onnx")
            torch.onnx.export(graph, args, str(target), dynamo=True, opset_version=18,
                              input_names=inputs, output_names=[output], dynamic_shapes=dynamic,
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
            try:
                session = ort.InferenceSession(str(target), providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise RuntimeError(f"ONNX Runtime cannot execute {filename}.onnx: {exc}") from exc
            for na, nf in [(1,1),(2,3),(3,2),(4,5)]:
                xx = x + 0.1*rand(nx,1)
                ww, vv, dd = rand(ny,na), rand(nx,nf), rand(ny,na*nf)
                values = {"x":xx,"adj_y":ww,"fwd_x":vv,"fwd_adj_y":dd}
                actual = session.run(None, {key:values[key].numpy() for key in inputs})[0]
                # Compare with forward AD, independent of the export workaround.
                if filename == "fwd_adj_"+name:
                    ref = torch.vmap(lambda vi, dwi: torch.func.jvp(adjoint,
                        (xx[:,0],ww.T),(vi,dwi))[1])(vv.T,dd.T.reshape(nf,na,ny)).reshape(-1,nx).T
                elif filename == "fwd_"+name:
                    ref = torch.vmap(lambda vi:torch.func.jvp(primal,(xx[:,0],),(vi,))[1])(vv.T).T
                else:
                    ref = fn(*(values[key] for key in inputs))
                ref = ref.detach().numpy()
                if not np.isfinite(actual).all() or not np.isfinite(ref).all():
                    raise ValueError(f"Non-finite values while validating {filename}.onnx")
                tolerance = 2e-5 if example_input.dtype == torch.float32 else 1e-8
                np.testing.assert_allclose(actual, ref, rtol=tolerance, atol=tolerance*0.1)
        for filename, *_ in specifications:
            (temporary/(filename+".onnx")).replace(directory/(filename+".onnx"))
    return directory/(name+".onnx")
