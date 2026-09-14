# Changelog

## 0.1.0.dev0

Initial development release:

- Export primal, adjoint and forward-over-adjoint ONNX graphs for CasADi.
- Optionally export forward sensitivities.
- Support fixed-shape float32/float64 tensor models with dynamic seed counts.
- Validate exported values and derivative products against PyTorch.
- Preserve caller model state and stage files until validation succeeds.

Requires PyTorch 2.6 and the pinned exporter dependencies. CasADi integration
requires the sibling derivative discovery feature described in the README.
