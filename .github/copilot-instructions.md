# Copilot Instructions for `nou` Project

## Project Overview

**nou** is a modern C++23 compile-time neural network library using C++ modules. It provides compile-time type-safe neural network functionality through concept-based design and customization point objects (CPOs). Return types are zero-overhead: layers may return `T` or `std::expected<T, computation_error>`, and `predict()` returns `T` only if every layer is error-free.

## Architecture & Core Concepts

### Module-Based Design

- **Single Module Entry**: [include/nou/nou.cppm](../include/nou/nou.cppm) - exports all public APIs
- **Key Concepts**: `shape<N>` (dimension descriptor), `layer` (layer interface), `network` (tuple-like container of layers)
- **Central Functions**: `make_network()` composes layers/builders; `predict()` propagates inputs and conditionally returns `T` or `std::expected<T, computation_error>`
- **CPO Pattern**: `build()` customization point supports member functions, free functions, and hidden friend functions; `propagate` CPO supports member or ADL free/hidden friend

### Type System Fundamentals

- **`shape<N>`**: Compile-time dimension size (integral auto, N > 0)
- **`layer` Concept**: Requires `real_type`, `input_shape_type`, `output_shape_type` member types
- **`network` Concept**: A `std::tuple` containing only `layer` types
- **`is_connectable`**: Validates adjacent layers/builders have matching input/output shapes
- **Error Model**: `computation_error` (uint8_t) is the unified runtime error; `computation_errorable` concept detects `std::expected<T, computation_error>`

### Design Patterns

1. **Concept Constraints**: Heavy use of `requires` clauses to enforce compile-time contracts
2. **Type Traits**: `is_network<T>` specialization for tuple detection
3. **Customization Points**: `build()` customization point allows multiple builder patterns
4. **Builder Factories**: Convert builders to layers dynamically based on previous layer's real_type and output_shape

## Development Workflows

### Build & Test (local)

```bash
# Configure with Ninja and warnings-as-errors (C++23)
cmake -Bbuild -GNinja -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Debug

# Build (uses C++23 modules)
cmake --build build --config Debug -j4 --verbose

# Run tests (single executable per test)
ctest --test-dir build --verbose

# Lint (clang-tidy using checks configured in the repository .clang-tidy file)
run-clang-tidy-19 -p build include
```

### Test Framework

- **Framework**: Tests use `static_assert` for compile-time verification within standard C++ blocks
- **Test Structure**: [test/unit_test/](../test/unit_test/) contains `.cc` files
- **Test Registration**: `nou_test(name)` CMake function auto-registers executables
- **Concept Testing**: Tests heavily use compile-time assertions (`static_assert`)

## Code Conventions & Patterns

### C++23 Module Patterns

- Use `export module nou;` for public exports
- Internal helpers use unnamed namespace, not `export`
- Template specializations stay internal unless exported
- Exported APIs include `make_network`, `predict`, `build` CPO, `propagate` CPO, and `computation_error`

### Builder Implementation Options

All three patterns are tested and supported:
```cpp
// 1. Member function
struct MyBuilder {
  template <class Real, class Shape>
  auto build() const -> layer;
};

// 2. Free function (ADL via nou namespace)
template <class Real, class Shape>
auto build(const MyBuilder& b) -> layer;

// 3. Hidden friend
struct MyBuilder {
  template <class Real, class Shape>
  friend auto build(const MyBuilder& b) -> layer;
};
```

### Test Pattern

- Use `compile_fail_*` concepts to verify rejection of invalid types
- Use `static_assert` for compile-time verification (no runtime assertions)
- Layer test structs always include `.value` field for equality testing
- Tests use standard C++ blocks with comments for test names (e.g., `// --- test name --- \n { ... }`)

## Integration Points & Dependencies

### External Dependencies

- **No External Libraries**: All core functionality is implemented from scratch using only the C++23 standard library
- **Standard Library**: C++23 standard library features; `std::expected` required

### CMake Configuration

- **Language**: C++23 required, enforced by CMakeLists.txt
- **Compiler Warnings**: All warnings as errors (`-Werror` on GCC/Clang, `/WX` on MSVC)
- **Clang-Tidy**: Enabled in CI for clang++-19; specific check categories enabled (bugprone, cert, cppcoreguidelines, clang-analyzer, google, misc, modernize, performance, readability) with warnings-as-errors
- **Modules**: Uses CMake FILE_SET CXX_MODULES target property
- **Known clang-tidy false positive**: `misc-use-internal-linkage` on exported functions in the module; suppressed with `// NOLINTNEXTLINE(misc-use-internal-linkage)`

### CI/CD

- **Linux CI** (.github/workflows/linux.yml): Tests g++-14 and clang++-19
- **Linting**: clang++-19 job runs clang-tidy-19

## Critical Developer Knowledge

### Shape Matching Validation

The `is_connectable` trait recursively validates that:
- First argument is a `layer`
- Each subsequent argument is either: a `layer` with matching input_shape, OR a builder that can produce one
- Output shape of layer N = input shape of layer N+1

### Why Compile-Time?

All computations are `constexpr` to ensure:
- Zero runtime overhead
- Type safety at compile time
- Early error detection during compilation
- Conditional error handling at zero cost when layers return plain `T` (no `std::expected` path taken)

### Adding New Tests

1. Create [test/unit_test/new_test.cc](../test/unit_test/)
2. Use `import nou;` and standard includes
3. Add `nou_test(new_test)` to [test/unit_test/CMakeLists.txt](../test/unit_test/CMakeLists.txt)
4. Use `static_assert` for compile-time checks and `compile_fail_*` concepts for rejection validation
5. Use standard C++ blocks with comment headers for test names (e.g., `// --- test name --- \n { ... }`)

## Format & Style

- **Clang-format**: [.clang-format](../.clang-format) enforces 80-column limit, 2-space indent
- **Clang-tidy**: [.clang-tidy](../.clang-tidy) enforces modern C++, no NOLINT comments without reason
- **Comment Style**: Detailed Doxygen comments on exported types and functions
