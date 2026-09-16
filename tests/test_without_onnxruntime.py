import subprocess
import sys
import textwrap
import unittest


class RuntimeDependencyTests(unittest.TestCase):
    def test_export_without_onnxruntime(self):
        code = textwrap.dedent('''
            import sys
            sys.modules['onnxruntime'] = None
            import tempfile
            from pathlib import Path
            import torch
            from torch2casadi import export
            torch.set_num_threads(1)
            with tempfile.TemporaryDirectory() as directory:
                export(torch.nn.Tanh(), torch.zeros(2), directory, forward=True)
                assert {p.name for p in Path(directory).iterdir()} == {
                    'f.onnx', 'adj_f.onnx', 'fwd_adj_f.onnx', 'fwd_f.onnx'}
        ''')
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
