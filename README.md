# torch2casadi

Export a PyTorch model and its derivative graphs to ONNX, then use it in CasADi optimization. PyTorch is needed at export time; evaluation uses CasADi's native ONNX Runtime backend.

This is a local, unpublished prototype intended for `casadi/torch2casadi`. It is not yet available from PyPI.

```python
import torch
import casadi as ca
from torch2casadi import export

model = torch.nn.Sequential(
    torch.nn.Linear(2, 16), torch.nn.Tanh(), torch.nn.Linear(16, 2)
).eval()
path = export(model, torch.zeros(1, 2), "generated")
f = ca.GraphBuilder(str(path)).create("f")

opti = ca.Opti()
u = opti.variable(2)
opti.minimize(ca.sumsqr(f(u) - ca.DM([0.9, -0.2])))
opti.subject_to(opti.bounded(-1, u, 1))
opti.solver("sqpmethod", {"qpsol": "qrqp"})
solution = opti.solve()
```

The trained-surrogate example in `examples/opti.py` includes a coupled constraint and regularization. It uses exact Hessians.

## Installation

Use Python 3.10 or later and install the local checkout with `python -m pip install .`. Install the appropriate PyTorch 2.6 CPU/CUDA wheel first if necessary. The dependency versions in `pyproject.toml` describe the tested exporter stack; they are deliberately narrow for this initial prototype.

For CasADi integration, build branch `onnx-primal-efficiency` with `WITH_ONNX=ON` and `WITH_ONNX_RUNTIME=ON`. The sibling derivative feature is in commit `b5040ffd76`. Installing the Python `onnxruntime` wheel does not enable CasADi's native backend. The exporter itself does not depend on the CasADi Python package.

## Export contract

- One tensor input and one tensor output; float32 or float64 on CPU during export.
- Tensor shapes are fixed by the example. All elements are differentiable, including any model batch axes. No independence between batch elements is assumed.
- The module is copied, switched to evaluation mode and frozen. The caller's parameters and training mode are preserved.
- Inputs and outputs are flattened in PyTorch element order and exposed as CasADi column vectors. Derivative direction counts are dynamic.
- Modules must support the required PyTorch function transforms and ONNX operations. Data-dependent Python branches, stochastic evaluation, integer/discrete inputs, multiple argument/output trees, and unsupported custom operators are outside this initial contract.
- Smooth models are appropriate for exact-Hessian optimization. ReLU networks do not acquire smooth Hessians through export.

Default files:

| File | Inputs | Output |
| --- | --- | --- |
| `f.onnx` | `x`: nx × 1 | `y`: ny × 1 |
| `adj_f.onnx` | `x`, `adj_y`: ny × nadj | `adj_x`: nx × nadj |
| `fwd_adj_f.onnx` | `x`, `adj_y`, `fwd_x`: nx × nfwd, `fwd_adj_y`: ny × (nfwd·nadj) | `fwd_adj_x`: nx × (nfwd·nadj) |

`export(..., forward=True)` also writes `fwd_f.onnx`. `name="net"` changes the family to `net.onnx`, `adj_net.onnx`, etc.

Combined columns are grouped by outer forward direction, then inner adjoint direction. The mixed graph differentiates both the primal input and the adjoint seed. It computes products through AD without forming a dense Jacobian or Hessian as an intermediate.

Each export checks all graphs with ONNX's checker and executes them with independent `(nadj, nfwd)` counts `(1,1)`, `(2,3)`, `(3,2)`, `(4,5)` at perturbed inputs. Mixed and forward derivatives are checked against PyTorch forward AD, independently of the reverse-over-reverse export formulation. Files are staged until all checks pass. Export into a fresh model-family directory, so old derivative files cannot silently survive a changed model. Validation uses relative/absolute tolerances of 2e-5/2e-6 for float32 and 1e-8/1e-9 for float64; ORT fusion can round scalar coefficients even in a double graph. These numerical checks supplement the exporter contract; they cannot prove correctness over arbitrary data-dependent behavior.

Keep sibling files together until derivatives are constructed. Constructed CasADi derivatives embed their model bytes, including when serialized.

## Development

With the feature CasADi build on `PYTHONPATH`:

```sh
python -m unittest discover -s tests
python -m build
```

The tests cover Tanh, Sigmoid, Softplus, tensor-shaped nonlinear algebra, independent seed counts, Hessians through CasADi and serialization after the ONNX files are deleted.

The implementation uses experimental FX tracing APIs and a PyTorch 2.6 export workaround. Broader PyTorch-version/operator support needs a compatibility test matrix before a stable release. Future arbitrary derivative-family export can build on the same packaging convention; this initial package emits first derivatives and forward-over-adjoint, not all nesting patterns supported by CasADi.
