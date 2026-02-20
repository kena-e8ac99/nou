# nou - C++23 Neural Network Library

> ⚠️ **Personal Learning Project - Under Development**
>
> This project is a personal learning initiative and is currently under development.
> The API may change significantly, and features are not yet stable. Use at your own risk.

## Overview

`nou` is a modern, compile-time neural network library written in C++23 with modules support. It provides a type-safe and efficient foundation for neural network functionality through static polymorphism and modern C++ features.

The library is designed with the following goals:

- **Compile-time computation**: Leverage C++23 features for static polymorphism and type safety
- **Modular architecture**: Clean separation of concerns across specialized modules
- **No external dependencies**: Built only with the C++ standard library
- **Type safety**: Use concepts and `requires` clauses for compile-time verification

## Modules

The library is organized into specialized modules:

### `nou.core`

Core neural network building functionality:

- **Concepts**: `layer`, `sequential_network` (using `std::extents` for shape representation)
- **Types**: Network building utilities
- **CPOs**: `build` - customization point object for network construction

### `nou.predict`

Prediction and inference functionality:

- **Concepts**:
  - Input/Output validation: `input`, `output`, `batch_input`, `batch_output`
  - Shape compatibility: `compatible_extents_with`
  - Function concepts: `mathematical_function`, `differentiable_function`
- **Types**: `computation_error` for error handling
- **Functions**: `predict()` for single and batch predictions
- **CPOs**: `propagate` for forward propagation

### `nou.evaluate`

Evaluation and metrics functionality:

- **Functions**: `evaluate()` for model assessment (single and batch versions)

### `nou` (Umbrella Module)

Re-exports all public functionality from the above modules.

## Building

### Requirements

- C++23-capable compiler
  - GCC 13.1+
  - Clang 17+ (with `-stdlib=libc++` and `-fexperimental-library`)
  - MSVC 2022+
- CMake 3.28+

### Build Instructions

```bash
# Configure the project
cmake -B build -S . -DCMAKE_CXX_STANDARD=23

# Build the library
cmake --build build

# Run tests (if top-level project)
ctest --test-dir build
```

### Using with libc++

For Clang with libc++ backend:

```bash
cmake -B build -S . \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=23
cmake --build build
```

## Testing

The project includes comprehensive unit tests using compile-time assertions where possible and runtime assertions for validation:

```bash
# Run all tests
cmake --build build --target test

# Or using CTest
ctest --test-dir build --verbose
```

Test files are located in [test/unit_test/](test/unit_test/) and cover:

- Network creation and building
- Prediction (single and batch)
- Evaluation metrics
- Model fitting

## Project Structure

```
.
├── CMakeLists.txt              # Main CMake configuration
├── include/
│   └── nou/
│       ├── core.cppm           # Core module interface
│       ├── predict.cppm        # Prediction module interface
│       ├── evaluate.cppm       # Evaluation module interface
│       └── nou.cppm            # Umbrella module
├── test/
│   └── unit_test/
│       ├── ut_make_network.cc  # Network creation tests
│       ├── ut_prediction.cc    # Prediction tests
│       ├── ut_evaluate.cc      # Evaluation tests
│       └── ut_fit.cc           # Model fitting tests
└── README.md                   # This file
```

## Design Principles

1. **Type Safety**: Extensive use of concepts and `requires` clauses to enforce constraints at compile-time
2. **Shape Representation**: Uses `std::extents` (std::mdspan-compatible) for flexible shape specification
3. **Error Handling**: Employs `std::expected<T, computation_error>` for fallible operations
4. **Minimal API**: Public API is kept minimal; only necessary functionality is exported
5. **No Circular Dependencies**: Module structure strictly avoids circular imports

## Code Quality

The project maintains strict code quality standards:

- Compile-time checks with `-Wall -Wextra -Wpedantic -Werror -Wshadow` (GCC/Clang)
- Comprehensive static assertions for compile-time validation
- Runtime assertions for runtime verification
- Doxygen-style documentation for all public API

## Development

When contributing or extending this library, please adhere to:

- Use compile-time `static_assert` tests wherever possible
- Use `assert` for runtime tests (no external testing frameworks)
- Follow standard library wording and conventions in documentation
- Keep modules free of circular dependencies
- Use `std::extents` for all shape specifications

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.
