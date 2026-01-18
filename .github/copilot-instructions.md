# Copilot Instructions for `nou` Project

## What this project is
`nou` is a C++23 compile-time neural network library built with modules. It aims to provide a modern, efficient, and type-safe neural network functionality.

## Modules (names and roles)
- `nou.core` ([include/nou/core.cppm](../include/nou/core.cppm))
  - Provide core neural network building functionality:
    - Concepts: `layer`, `sequential_network`, utility concepts for connectivity; uses `std::extents` for shape representation
    - Free functions: `make_network`
    - CPOs: `build`
- `nou.predict` ([include/nou/predict.cppm](../include/nou/predict.cppm))
  - Provide prediction-related functionality:
    - Types: `computation_error`
    - Concepts:
      - errorable output: `computation_errorable`
      - input and output concepts for both single and batch versions: `input`, `output`, `batch_input`, `batch_output` and other utilities
      - mathematical function concepts: `mathematical_function`, `differentiable_function`
    - Free functions: `predict` (single and batch versions)
    - CPOs: `propagate` (single and batch versions)
- `nou.evaluate` ([include/nou/evaluate.cppm](../include/nou/evaluate.cppm))
  - Provide evaluation-related functionality:
    - Free functions: `evaluate` (single and batch versions)
- `nou` ([include/nou/nou.cppm](../include/nou/nou.cppm))
  - Umbrella module re-exporting all functionality

## Constraints the assistant must respect
- Modules:
  - Do not introduce circular dependencies
  - Keep public API minimal; only export what is necessary
- Types:
  - Use `std::extents` (no custom shape types); layers expose `real_type`, `input_extents`, `output_extents`
- Concepts:
  - Distinguish single vs batch input/output concepts
  - Use `requires` clauses to enforce constraints, avoid SFINAE
- Error handling:
  - Use `std::expected<T, computation_error>` if and only if fallible operation. Otherwise, use direct types
- Testing (in [test/](../test/)):
  - Use `static_assert` compile-time tests wherever possible. For runtime tests, use `assert`; do not use any testing frameworks
  - Separate tests by enclosing scope blocks with comment headers describing test name
  - Test naming pattern:
    - Use `// --- Test that [subject] [predicate] [conditions] ---`
      - subject: test target (function, concept, behavior)
      - predicate: expected behavior (e.g., "returns", "accepts", "rejects")
      - conditions: optional, any specific conditions
    - Wrap long names onto a second indented line (only last line ends with `---`)
- Documentation:
  - Follow standard-library wording and style
  - Use Doxygen style comments for all public API
