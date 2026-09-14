import tempfile
import unittest
from pathlib import Path

import os
if os.environ.get("TORCH2CASADI_INTEGRATION") == "1":
    import casadi as ca
import numpy as np
import torch
from torch import nn
from torch2casadi import export


class MatrixModel(nn.Module):
    def forward(self, x):
        return torch.sin(x) + 0.2 * (x @ x.T)


@unittest.skipUnless(os.environ.get("TORCH2CASADI_INTEGRATION") == "1",
                     "set TORCH2CASADI_INTEGRATION=1 with the feature CasADi build")
class CasadiIntegrationTests(unittest.TestCase):
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
                f = ca.Function.deserialize(ca.GraphBuilder(str(path)).create("f").serialize())
                nx, ny = sample.numel(), model(sample).numel()
                X = ca.MX.sym("x",nx)
                weights = torch.linspace(0.5,1.5,ny,dtype=sample.dtype)
                H = ca.Function("H",[X],[ca.hessian(ca.dot(ca.DM(weights.numpy()),f(X)),X)[0]])
                for point in [torch.ones_like(sample)*0.2,torch.ones_like(sample)*-0.3]:
                    fun = lambda x:(model(x.reshape(sample.shape)).reshape(-1)*weights).sum()
                    expected = torch.func.hessian(fun)(point.reshape(-1)).detach().numpy()
                    np.testing.assert_allclose(np.array(H(point.reshape(-1).numpy())),expected,rtol=2e-4,atol=2e-6)
                    for na,nf in [(1,1),(2,3),(3,2)]:
                        xx=point.reshape(-1)
                        w=torch.randn(ny,na,dtype=sample.dtype)
                        v=torch.randn(nx,nf,dtype=sample.dtype)
                        dw=torch.randn(ny,na*nf,dtype=sample.dtype)
                        primal=lambda x:model(x.reshape(sample.shape)).reshape(-1)
                        a=lambda x,w:torch.vmap(torch.func.vjp(primal,x)[1])(w)[0]
                        ref=torch.vmap(lambda vi,dwi:torch.func.jvp(a,(xx,w.T),(vi,dwi))[1])(
                            v.T,dw.T.reshape(nf,na,ny)).reshape(-1,nx).T.detach().numpy()
                        adj=f.reverse(na)
                        y=f(xx.numpy()); av=adj(xx.numpy(),y,w.numpy())
                        actual=adj.forward(nf)(xx.numpy(),y,w.numpy(),av,v.numpy(),np.zeros((ny,nf)),dw.numpy())
                        np.testing.assert_allclose(np.array(actual),ref,rtol=2e-4,atol=2e-6)
                serialized=H.serialize()
                for file in Path(d).glob('*.onnx'):
                    file.unlink()
                np.testing.assert_allclose(np.array(ca.Function.deserialize(serialized)(point.reshape(-1).numpy())),expected,rtol=2e-4,atol=2e-6)

if __name__ == '__main__':
    unittest.main()
