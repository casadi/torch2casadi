# Changelog

## 0.1.2

- Move ONNX Runtime numerical validation into tests; exports retain ONNX structural checks.
- Add `overwrite=True` to replace validated model families and remove stale derivatives.
- Accept multiple tensor arguments and per-input `is_diff_in` flags. Runtime parameters
  remain graph inputs, but their adjoints and forward seeds are omitted.
- Add a four-stage FATROP shuttle example with a temperature-dependent NN path constraint.

## 0.1.1

- Remove all runtime dependency declarations so installation preserves user-managed
  Torch builds and other packages. Keep the tested export environment in CI only.

## 0.1

Initial release:

- Export primal, adjoint and forward-over-adjoint ONNX graphs for CasADi.
- Optionally export forward sensitivities.
- Support fixed-shape float32/float64 tensor models with dynamic seed counts.
- Validate exported values and derivative products against PyTorch.
- Preserve caller model state and stage files until validation succeeds.

Requires PyTorch 2.6 and the pinned exporter dependencies. CasADi integration
requires the sibling derivative discovery feature described in the README.
