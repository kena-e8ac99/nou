# Copilot Instructions for `nou` Project

## Project Overview

**nou** is a modern C++20 compile-time neural network library using C++ modules. It provides compile-time type-safe neural network functionality through concept-based design and customization point objects (CPOs).

## Architecture & Core Concepts

### Module-Based Design

- **Single Module Entry**: [include/nou/nou.cppm](../include/nou/nou.cppm) - exports all public APIs
- **Key Concepts**: `shape<N>` (dimension descriptor), `layer` (layer interface), `network` (tuple-like container of layers)
- **Central Function**: `make_network()` - composes layers or builders into type-safe networks
- **CPO Pattern**: `build()` customization point supports member functions, free functions, and hidden friend functions

### Type System Fundamentals

- **`shape<N>`**: Compile-time dimension size (integral auto, N > 0)
- **`layer` Concept**: Requires `real_type`, `input_shape_type`, `output_shape_type` member types
- **`network` Concept**: A `std::tuple` containing only `layer` types
- **`is_connectable`**: Validates adjacent layers/builders have matching input/output shapes

### Design Patterns

1. **Concept Constraints**: Heavy use of `requires` clauses to enforce compile-time contracts
2. **Type Traits**: `is_network<T>` specialization for tuple detection
3. **Customization Points**: `build()` customization point allows multiple builder patterns
4. **Builder Factories**: Convert builders to layers dynamically based on previous layer's real_type and output_shape

## Development Workflows

### Build & Test

```bash
# Configure with Ninja and warnings-as-errors
cmake -Bbuild -GNinja -DCMAKE_CXX_COMPILER=clang++-18 -DCMAKE_BUILD_TYPE=Debug

# Build (uses C++20 modules)
cmake --build build --config Debug -j4 --verbose

# Run tests (single executable per test)
ctest --test-dir build --verbose

# Lint (clang-tidy using checks configured in the repository .clang-tidy file)
run-clang-tidy-18 -p build include
```

### Test Framework

- **Framework**: [Boost.UT](https://github.com/boost-ext/ut) (fetched via CMake)
- **Test Structure**: [test/unit_test/](../test/unit_test/) contains `.cc` files
- **Test Registration**: `nou_test(name)` CMake function auto-registers executables
- **Concept Testing**: Tests heavily use compile-time assertions (`static_assert`)

## Code Conventions & Patterns

### C++20 Module Patterns

- Use `export module nou;` for public exports
- Internal helpers use unnamed namespace, not `export`
- Template specializations stay internal unless exported

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

## Integration Points & Dependencies

### External Dependencies

- **No External Libraries other than test framework**: All core functionality is implemented from scratch
- **Boost.UT** (v2.3.1): Header-only test framework, fetched at configure time
- **Standard Library**: C++20 standard library features only

### CMake Configuration

- **Language**: C++20 required, enforced by CMakeLists.txt
- **Compiler Warnings**: All warnings as errors (`-Werror` on GCC/Clang, `/WX` on MSVC)
- **Clang-Tidy**: Enabled in CI for clang++-18; all checks enabled with warnings-as-errors
- **Modules**: Uses CMake FILE_SET CXX_MODULES target property

### CI/CD

- **Linux CI** (.github/workflows/linux.yml): Tests g++-14 and clang++-18
- **Linting**: Only clang++-18 runs clang-tidy to save CI time

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

### Adding New Tests

1. Create [test/unit_test/new_test.cc](../test/unit_test/)
2. Use `#include <boost/ut.hpp>` and `import nou;`
3. Add `nou_test(new_test)` to [test/unit_test/CMakeLists.txt](../test/unit_test/CMakeLists.txt)
4. Use `static_assert` for compile-time checks and `compile_fail_*` concepts for rejection validation

## Format & Style

- **Clang-format**: [.clang-format](../.clang-format) enforces 80-column limit, 2-space indent
- **Clang-tidy**: [.clang-tidy](../.clang-tidy) enforces modern C++, no NOLINT comments without reason
- **Comment Style**: Detailed Doxygen comments on exported types and functions
