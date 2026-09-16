"""Test exported graphs against independent PyTorch AD using ONNX Runtime."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch2casadi import export as export_onnx


def export(model, example_input, directory, **options):
    path = export_onnx(model, example_input, directory, **options)
    model = deepcopy(model).cpu().eval().requires_grad_(False)
    samples = example_input if isinstance(example_input, tuple) else (example_input,)
    samples = tuple(x.detach().cpu() for x in samples)
    names = options.get('input_names') or (['x'] if len(samples) == 1 else
                                          [f'x{i}' for i in range(len(samples))])
    active = tuple(i for i, enabled in enumerate(options.get('is_diff_in', [True]*len(samples)))
                   if enabled)
    name = options.get('name', 'f')
    generator = torch.Generator().manual_seed(2026)

    def rand(*shape):
        return torch.randn(shape, generator=generator, dtype=samples[0].dtype)

    families = [name]
    if active:
        families += ['adj_'+name, 'fwd_adj_'+name]
        if options.get('forward'):
            families += ['fwd_'+name]
    sessions = {key: ort.InferenceSession(str(Path(directory)/(key+'.onnx')),
                providers=['CPUExecutionProvider']) for key in families}
    for na, nf in [(1, 1), (2, 3), (3, 2), (4, 5)]:
        xs = tuple(x.reshape(-1)+0.1*rand(x.numel()) for x in samples)

        def primal(*args):
            values = list(xs)
            for i, value in zip(active, args):
                values[i] = value
            return model(*(value.reshape(sample.shape) for value, sample
                           in zip(values, samples))).reshape(-1)

        point = tuple(xs[i] for i in active)
        y = primal(*point)
        values = {key: x[:, None] for key, x in zip(names, xs)}
        expected = {name: (y[:, None],)}
        if active:
            w = rand(y.numel(), na)
            directions = tuple(rand(xs[i].numel(), nf) for i in active)
            dw = rand(y.numel(), na*nf)
            values['adj_y'] = w
            values['fwd_adj_y'] = dw
            values.update({'fwd_'+names[i]: v for i, v in zip(active, directions)})

            def adjoint(*args):
                *inputs, weights = args
                pullback = torch.func.vjp(primal, *inputs)[1]
                columns = [pullback(weights[:, k]) for k in range(na)]
                return tuple(torch.stack([col[i] for col in columns], dim=1)
                             for i in range(len(active)))

            expected['adj_'+name] = adjoint(*point, w)
            mixed = [torch.func.jvp(adjoint, point+(w,),
                     tuple(v[:, j] for v in directions)+(dw[:, j*na:(j+1)*na],))[1]
                     for j in range(nf)]
            expected['fwd_adj_'+name] = tuple(torch.cat([v[i] for v in mixed], dim=1)
                                             for i in range(len(active)))
            if options.get('forward'):
                expected['fwd_'+name] = (torch.stack([
                    torch.func.jvp(primal, point, tuple(v[:, j] for v in directions))[1]
                    for j in range(nf)], dim=1),)
        for key, session in sessions.items():
            actual = session.run(None, {v.name: values[v.name].numpy()
                                       for v in session.get_inputs()})
            assert len(actual) == len(expected[key]), key
            for got, reference in zip(actual, expected[key]):
                reference = reference.detach().numpy()
                assert np.isfinite(got).all() and np.isfinite(reference).all(), key
                tolerance = 2e-5 if samples[0].dtype == torch.float32 else 1e-8
                np.testing.assert_allclose(got, reference, rtol=tolerance,
                                           atol=tolerance*0.1, err_msg=key)
    return path
