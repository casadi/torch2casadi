"""Fit a smooth PyTorch surrogate, export its derivatives, and optimize with Opti.

Requires torch2casadi and a CasADi build with sibling ONNX derivative discovery
(branch onnx-primal-efficiency), WITH_ONNX=ON and WITH_ONNX_RUNTIME=ON.
Run: python example.py --output-dir generated
"""
import argparse
from pathlib import Path

import casadi as ca
import numpy as np
import torch
from torch import nn
from torch2casadi import export


def make_model():
    torch.manual_seed(42)
    return nn.Sequential(nn.Linear(2, 16), nn.Tanh(), nn.Linear(16, 16),
                         nn.Tanh(), nn.Linear(16, 2))


def train_model():
    model = make_model()
    inputs = 2*torch.rand(512, 2)-1
    outputs = torch.stack((inputs[:, 0]+0.2*torch.sin(inputs[:, 1]),
                           inputs[:, 1]+0.1*inputs[:, 0]**2), dim=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for _ in range(600):
        optimizer.zero_grad()
        loss = (model(inputs)-outputs).square().mean()
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


def optimization(f):
    opti = ca.Opti()
    u = opti.variable(2)
    y = f(u)
    opti.minimize(ca.sumsqr(y-ca.DM([0.9, -0.2])) + 0.01*ca.sumsqr(u))
    opti.subject_to(opti.bounded(-1, u, 1))
    opti.subject_to(ca.sum1(u) <= 0.45)
    opti.set_initial(u, 0)
    opti.solver("sqpmethod", {
        "qpsol": "qrqp", "tol_pr": 1e-7, "tol_du": 1e-6,
        "print_header": False, "print_iteration": False,
        "print_status": False, "print_time": False,
        "qpsol_options": {"print_header": False, "print_iter": False},
    })
    return opti, u


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("generated"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = train_model()
    path = export(model, torch.zeros(1, 2), args.output_dir)
    torch.save(model.state_dict(), args.output_dir/"weights.pt")
    f = ca.GraphBuilder(str(path)).create("f")
    opti, u = optimization(f)
    solution = opti.solve()
    control = np.asarray(solution.value(u)).reshape(1, 2)
    prediction = model(torch.tensor(control, dtype=torch.float32)).numpy()
    np.testing.assert_allclose(np.array(f(control.ravel())).ravel(), prediction.ravel(), atol=2e-6)
    assert np.max(np.abs(control)) <= 1+1e-7 and control.sum() <= 0.45+1e-7
    print("u =", control.ravel())
    print("surrogate output =", prediction.ravel())
    print("objective =", solution.value(opti.f))
    print("SQP iterations =", solution.stats()["iter_count"])


if __name__ == "__main__":
    main()
