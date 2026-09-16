import tempfile
import unittest
from pathlib import Path

import onnx
import torch
from torch import nn
from numerical_validation import export


class MatrixModel(nn.Module):
    def forward(self, x):
        return torch.sin(x) + 0.2 * (x @ x.T)


class ExportTests(unittest.TestCase):
    def test_models(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        cases = [
            (nn.Linear(2,2).double(), torch.zeros(2,dtype=torch.double)),
            (nn.Sequential(nn.Linear(4,8),nn.Tanh(),nn.Linear(8,2)).double(), torch.zeros(4,dtype=torch.double)),
            (nn.Sequential(nn.Linear(3,6),nn.Sigmoid(),nn.Linear(6,1)).double(), torch.zeros(1,3,dtype=torch.double)),
            (nn.Sequential(nn.Linear(3,5),nn.Softplus(),nn.Linear(5,2)), torch.zeros(1,3)),
            (MatrixModel(), torch.zeros(2,2)),
        ]
        for model, sample in cases:
            with self.subTest(model=model), tempfile.TemporaryDirectory() as d:
                original = {k:v.clone() for k,v in model.state_dict().items()}
                path = export(model, sample, d, forward=True)
                self.assertTrue(model.training)
                for k,v in original.items():
                    torch.testing.assert_close(v,model.state_dict()[k])
                self.assertEqual(path.name, "f.onnx")
                self.assertEqual({p.name for p in Path(d).iterdir()},
                                 {"f.onnx", "adj_f.onnx", "fwd_adj_f.onnx", "fwd_f.onnx"})
                for file in Path(d).glob("*.onnx"):
                    graph = onnx.load(file)
                    onnx.checker.check_model(graph)
                    self.assertFalse(any(i.external_data for i in graph.graph.initializer))

    def test_reject_invalid_input_and_existing_family(self):
        model = nn.Linear(2,2)
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(TypeError):
                export(model, torch.zeros(2,dtype=torch.int64),d)
            marker = Path(d)/"adj_f.onnx"
            marker.write_bytes(b"existing model")
            with self.assertRaises(FileExistsError):
                export(model, torch.zeros(2),d)
            self.assertEqual(marker.read_bytes(), b"existing model")

    def test_overwrite_family(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            paths = [Path(d)/name for name in ("f.onnx", "adj_f.onnx", "fwd_f.onnx")]
            for path in paths:
                path.write_bytes(b"previous model")
            unrelated = Path(d)/"other_f.onnx"
            unrelated.write_bytes(b"unrelated")
            with self.assertRaises(ValueError):
                export(nn.Identity(), torch.zeros(0), d, overwrite=True)
            for path in paths:
                self.assertEqual(path.read_bytes(), b"previous model")
            export(nn.Tanh(), torch.zeros(2), d, overwrite=True)
            self.assertFalse(paths[-1].exists())
            self.assertEqual(unrelated.read_bytes(), b"unrelated")
            for name in ("f", "adj_f", "fwd_adj_f"):
                onnx.checker.check_model(onnx.load(Path(d)/(name+".onnx")))

    def test_named_default_family(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as d:
            path = export(nn.Linear(2,1), torch.zeros(2), d, name="net")
            self.assertEqual(path.name, "net.onnx")
            self.assertEqual({p.name for p in Path(d).iterdir()},
                             {"net.onnx", "adj_net.onnx", "fwd_adj_net.onnx"})

if __name__ == '__main__':
    unittest.main()
