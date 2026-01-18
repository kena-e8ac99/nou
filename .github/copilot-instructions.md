# Copilot Instructions for `nou` Project

## What this project is
- Modern C++23 compile-time neural network library built with modules
- Zero-overhead types: layers may return `T` or `std::expected<T, computation_error>`; `predict()` returns plain `T` only when all layers are error-free

## Modules (names and roles)
- `nou.core` ([include/nou/core.cppm](../include/nou/core.cppm)): concepts for `layer`, `sequential_network`, connectivity, `build` CPO, `make_network`, `first_layer_t`, `last_layer_t`; shapes use `std::extents`
- `nou.predict` ([include/nou/predict.cppm](../include/nou/predict.cppm)): re-exports core; adds `computation_error`, `computation_errorable`, `input`/`output`/`batch_input`/`batch_output` concepts, `propagate` CPO, `predict` (single + batch)
- `nou` ([include/nou/nou.cppm](../include/nou/nou.cppm)): umbrella re-export of `nou.predict`

## Constraints the assistant must respect
- Use `std::extents` (no custom shape types); layers expose `real_type`, `input_extents`, `output_extents`
- Enforce concepts via `requires`; prefer `std::expected<T, computation_error>` for fallible paths
- Modules: keep helpers internal (no accidental `export`); use `export import` for re-exports
- Avoid non-ASCII unless already present; 80-col, 2-space indent per `.clang-format`

## Build/testing hints (only what’s special)
- C++23, modules enabled; clang++ with libc++ may need `-fexperimental-library`
- Warnings as errors; clang-tidy enabled (see `.clang-tidy`)
- Tests live in [test/unit_test](../test/unit_test/); primarily `static_assert` compile-time tests

## Test naming pattern
- Comment headers: `// --- Test that <behavior> ---`, wrap long names onto a second indented line

## Common pitfalls to avoid
- Do not reintroduce custom `shape<N>`; always use `std::extents`
- Keep batch and single-input concepts distinct (`input` vs `batch_input`)
- Preserve `std::expected` error propagation semantics

## When adding APIs
- Provide concise Doxygen in standard-library wording
- Keep exports minimal; helper utilities stay internal
