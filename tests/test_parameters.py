import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
import torch
from torch import nn
from numerical_validation import export


class WeatherModel(nn.Module):
    def forward(self, x, weather):
        return (1+weather)*(x**2).sum(dim=1, keepdim=True)


class WeatherNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(3, 8), nn.Tanh(), nn.Linear(8, 1))

    def forward(self, x, weather):
        return self.network(torch.cat((x, weather), dim=1))


class ThreeInputModel(nn.Module):
    def forward(self, x, weather, z):
        return (1+weather)*(x.square().sum()+z.square().sum())


class ParameterExportTests(unittest.TestCase):
    def test_omitted_weather_derivatives(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            export(WeatherModel(), (torch.ones(1,2), torch.ones(1,1)), directory,
                   input_names=["x", "weather"], is_diff_in=[True, False], forward=True)
            signatures = {
                'f': (["x", "weather"], ["y"]),
                'adj_f': (["x", "weather", "adj_y"], ["adj_x"]),
                'fwd_adj_f': (["x", "weather", "adj_y", "fwd_x", "fwd_adj_y"], ["fwd_adj_x"]),
                'fwd_f': (["x", "weather", "fwd_x"], ["fwd_y"]),
            }
            for name, (inputs, outputs) in signatures.items():
                graph = onnx.load(Path(directory)/(name+'.onnx')).graph
                self.assertEqual([v.name for v in graph.input], inputs)
                self.assertEqual([v.name for v in graph.output], outputs)

    def test_concatenated_network_dynamic_directions(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory() as directory:
            export(WeatherNetwork(), (torch.zeros(1,2), torch.zeros(1,1)), directory,
                   input_names=["x", "weather"], is_diff_in=[True, False], forward=True)
            graph = onnx.load(Path(directory)/'fwd_adj_f.onnx').graph
            forward = next(v for v in graph.input if v.name == 'fwd_x')
            self.assertEqual(forward.type.tensor_type.shape.dim[1].dim_param, 'nfwd')

    def test_multiple_active_arguments(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            export(ThreeInputModel(), (torch.ones(2), torch.ones(1), torch.ones(3)), directory,
                   input_names=["x", "weather", "z"], is_diff_in=[True, False, True], forward=True)
            graph = onnx.load(Path(directory)/'adj_f.onnx').graph
            self.assertEqual([v.name for v in graph.output], ['adj_x', 'adj_z'])
            graph = onnx.load(Path(directory)/'fwd_adj_f.onnx').graph
            self.assertEqual([v.name for v in graph.output], ['fwd_adj_x', 'fwd_adj_z'])
            self.assertNotIn('fwd_weather', [v.name for v in graph.input])

    def test_all_inputs_nondifferentiable(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            export(WeatherModel(), (torch.ones(1,2), torch.ones(1,1)), directory,
                   is_diff_in=[False, False], forward=True)
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['f.onnx'])

    def test_invalid_masks_and_names(self):
        samples = (torch.ones(1,2), torch.ones(1,1))
        with tempfile.TemporaryDirectory() as directory:
            for options in [dict(is_diff_in=[True]), dict(is_diff_in=[True, 0]),
                            dict(input_names=['x', 'x']), dict(input_names=['x', 'adj_y'])]:
                with self.assertRaises(ValueError):
                    export(WeatherModel(), samples, directory, **options)


@unittest.skipUnless(os.environ.get('TORCH2CASADI_INTEGRATION') == '1', 'needs feature CasADi build')
class ParameterIntegrationTests(unittest.TestCase):
    def test_weather_changes_state_derivatives(self):
        import casadi as ca
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = export(WeatherModel(), (torch.ones(1,2), torch.ones(1,1)), directory,
                          input_names=['x', 'weather'], is_diff_in=[True, False], forward=True)
            with self.assertRaisesRegex(RuntimeError, "incomplete adj signature"):
                ca.GraphBuilder(str(path)).create('explicit', {'is_diff_in': [True, True]}).reverse(1)
            for symbolic in [False, True]:
                f = ca.GraphBuilder(str(path)).create('f',
                    ({'symbolic': True, 'is_diff_in': [True, False]} if symbolic else {}))
                self.assertEqual(f.is_diff_in(), [True, False])
                x, weather = ca.MX.sym('x',2), ca.MX.sym('weather')
                y = f(x, weather)
                D = ca.Function('D', [x, weather], [y, ca.jacobian(y,x),
                    ca.hessian(y,x)[0], ca.jacobian(y,weather)])
                self.assertEqual(D.sparsity_out(3).nnz(), 0)
                rev = f.reverse(2)
                self.assertFalse(rev.is_diff_in(1))
                self.assertFalse(rev.is_diff_out(1))
                for p in [0., .4, 1.]:
                    values = D([.2,-.3], p)
                    for got, expected in zip(values, [(1+p)*.13,
                            (1+p)*np.array([[.4,-.6]]), 2*(1+p)*np.eye(2), 0.]):
                        np.testing.assert_allclose(got, expected, rtol=2e-5, atol=2e-6)
                    r = rev([.2,-.3], p, f([.2,-.3],p), ca.DM([[1,2]]))
                    np.testing.assert_allclose(r[0], (1+p)*np.array([[.4,.8],[-.6,-1.2]]), atol=2e-6)
                    np.testing.assert_allclose(r[1], 0., atol=0.)
                serialized = D.serialize()
                if symbolic:
                    for file in Path(directory).glob('*.onnx'):
                        file.unlink()
                np.testing.assert_allclose(ca.Function.deserialize(serialized)([.2,-.3],.4)[2],
                    2.8*np.eye(2), rtol=2e-5, atol=2e-6)




class FreeWeights(nn.Module):
    """Expose a module's parameters as forward arguments (weights as NLP variables)."""
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.names = [n for n, _ in model.named_parameters()]

    def forward(self, x, *params):
        return torch.func.functional_call(self.model, dict(zip(self.names, params)), (x,))


class WeightInputTests(unittest.TestCase):
    def test_weights_as_inputs(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1)).double()
        params = tuple(p.detach() for p in model.parameters())
        names = ["x", "W1", "b1", "W2", "b2"]
        with tempfile.TemporaryDirectory() as directory:
            export(FreeWeights(model), (torch.zeros(1, 2, dtype=torch.float64),)+params,
                   directory, input_names=names)
            graph = onnx.load(Path(directory)/'fwd_adj_f.onnx').graph
            self.assertEqual([v.name for v in graph.input], names+["adj_y"]+
                             ["fwd_"+n for n in names]+["fwd_adj_y"])
            self.assertEqual([v.name for v in graph.output], ["fwd_adj_"+n for n in names])

    def test_hessian_opt_out(self):
        torch.set_num_threads(1)
        torch.manual_seed(1)
        model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
        params = tuple(p.detach() for p in model.parameters())
        with tempfile.TemporaryDirectory() as directory:
            export(FreeWeights(model), (torch.zeros(1, 2),)+params, directory,
                   input_names=["x", "W1", "b1", "W2", "b2"], hessian=False)
            files = sorted(p.name for p in Path(directory).glob('*.onnx'))
            self.assertEqual(files, ['adj_f.onnx', 'f.onnx'])
            graph = onnx.load(Path(directory)/'adj_f.onnx').graph
            self.assertEqual([v.name for v in graph.output],
                             ['adj_x', 'adj_W1', 'adj_b1', 'adj_W2', 'adj_b2'])

if __name__ == '__main__':
    unittest.main()
