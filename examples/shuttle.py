"""Four-stage shuttle with a prescribed temperature input to a vibration NN.

Run with the feature CasADi build: python examples/shuttle.py --directory generated-shuttle
The scalar temperature is a runtime parameter, not an AD input.
"""
import argparse
from pathlib import Path

import casadi as ca
import numpy as np
import torch
from torch import nn
from torch2casadi import export


class Vibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 16), nn.Tanh(), nn.Linear(16, 1))

    def forward(self, x, temperature):
        return self.net(torch.cat((x, temperature), dim=1))


def train():
    torch.manual_seed(42)
    model = Vibration()
    data = torch.rand(1536, 3)*torch.tensor([3., 1.5, 2.])-torch.tensor([0., 0., 1.])
    target = (.3+.2*data[:, 0:1]+.04*data[:, 2:3])*data[:, 1:2]**2
    optimizer = torch.optim.Adam(model.parameters(), lr=.01)
    for _ in range(2500):
        optimizer.zero_grad()
        loss = (model(data[:, :2], data[:, 2:])-target).square().mean()
        loss.backward()
        optimizer.step()
    return model.eval().requires_grad_(False)


def problem(vibration):
    N, nx, nu = 4, 2, 1
    opti = ca.Opti()
    temperature = opti.parameter()
    x, a = ca.MX.sym("x", nx), ca.MX.sym("a", nu)
    F = ca.Function("F", [x, a], [ca.vertcat(x[0]+x[1]+.5*a, x[1]+a)])
    Xk = opti.variable(nx)
    Xs, Us = [Xk], []
    for k in range(N):
        Uk, Xk_plus_1 = opti.variable(nu), opti.variable(nx)
        opti.subject_to(Xk_plus_1 == F(Xk, Uk))
        if k == 0:
            opti.subject_to(Xk == [0, 0])
        opti.subject_to(vibration(Xk, temperature) <= .8)
        opti.subject_to(opti.bounded(-1, Uk, 1))
        opti.subject_to(opti.bounded(0, Xk[1], 1.5))
        Xs.append(Xk_plus_1)
        Us.append(Uk)
        Xk = Xk_plus_1
    opti.subject_to(Xk == [3, 0])
    X, U = ca.horzcat(*Xs), ca.horzcat(*Us)
    opti.minimize(ca.sumsqr(U))
    opti.set_initial(X, ca.DM([[0, .5, 1.5, 2.5, 3], [0, 1, 1, 1, 0]]))
    opti.set_initial(U, ca.DM([[1, 0, 0, -1]]))
    opti.solver("fatrop", {"structure_detection": "auto", "print_time": False,
                          "fatrop": {"print_level": 0, "tol": 1e-6}})
    return opti, X, U, temperature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("generated-shuttle"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    model = train()
    path = export(model, (torch.zeros(1, 2), torch.zeros(1, 1)), args.directory,
                  input_names=["x", "temperature"], is_diff_in=[True, False])
    results = {}
    for symbolic in [False, True]:
        vibration = ca.GraphBuilder(str(path)).create("vibration",
            {"symbolic": symbolic, "is_diff_in": [True, False]})
        opti, X, U, temperature = problem(vibration)
        for value in [0., 1.]:
            opti.set_value(temperature, value)
            sol = opti.solve()
            states, controls = np.array(sol.value(X)), np.array(sol.value(U))
            risk = model(torch.tensor(states[:, :-1].T, dtype=torch.float32),
                         torch.full((4, 1), value)).numpy().ravel()
            assert risk.max() <= .8+2e-6
            np.testing.assert_allclose(states[:, [0, -1]], [[0, 3], [0, 0]], atol=2e-6)
            results[symbolic, value] = controls
            print("symbolic =", symbolic, "temperature =", value,
                  "controls =", controls, "peak vibration =", risk.max(),
                  "objective =", sol.value(opti.f))
    for value in [0., 1.]:
        np.testing.assert_allclose(results[False, value], results[True, value], atol=2e-5)
    assert np.max(np.abs(results[True, 0.]-results[True, 1.])) > .001
    print("Numeric/symbolic equivalence and temperature dependence passed.")


if __name__ == '__main__':
    main()
