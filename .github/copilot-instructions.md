# Copilot Instructions for `nou` Project

## Project Overview

**nou** is a modern C++23 compile-time neural network library using C++ modules. It provides compile-time type-safe neural network functionality through concept-based design and customization point objects (CPOs). Return types are zero-overhead: layers may return `T` or `std::expected<T, computation_error>`, and `predict()` returns `T` only if every layer is error-free.

## Architecture & Core Concepts

### Module-Based Design

The library uses a modular architecture with clear separation of concerns:

- **`nou.core`** ([include/nou/core.cppm](../include/nou/core.cppm)) - Core concepts and network construction
  - Exports: `layer` concept, `sequential_network` concept, `is_connectable` trait
  - Exports: `build` CPO, `make_network()` function
  - Exports: `first_layer_t<Network>`, `last_layer_t<Network>` type aliases
  - Uses `std::extents` for shape representation (no custom `shape<N>` type)
  - No dependencies on other nou modules
  
- **`nou.predict`** ([include/nou/predict.cppm](../include/nou/predict.cppm)) - Inference pipeline and error handling
  - Re-exports: `nou.core` (provides transitive access to core types)
  - Exports: `computation_error` enum, `computation_errorable` concept
  - Exports: `input<Input, Layer>`, `output<Output, Layer>` concepts for single-sample validation
  - Exports: `batch_input<BatchInput, Layer>`, `batch_output<BatchOutput, Layer>` concepts for batch validation
  - Exports: `propagate` CPO (supports both single-sample and batch inputs)
  - Exports: `predict()` function overloads for single-sample and batch inference
  
- **`nou`** ([include/nou/nou.cppm](../include/nou/nou.cppm)) - Umbrella module for convenience
  - Re-exports: `nou.predict` (transitively includes `nou.core`)
  - Single-import entry point: `import nou;` provides access to all functionality

**Module Dependency Chain**: `nou.core` → `nou.predict` → `nou` (umbrella)

**Key Concepts**: `layer` (layer interface with `std::extents` for shapes), `sequential_network` (tuple of connectable layers), `input`/`output` (single-sample type validation), `batch_input`/`batch_output` (batch type validation)

**Central Functions**: `make_network()` composes layers/builders; `predict()` propagates single-sample or batch inputs and conditionally returns `T` or `std::expected<T, computation_error>`

**CPO Pattern**: `build()` customization point supports member functions, free functions, and hidden friend functions; `propagate` CPO supports member or ADL free/hidden friend for both single and batch inputs

### Type System Fundamentals

- **`std::extents`**: Standard library type for compile-time shape representation; layers use `std::extents<std::size_t, N>` for `input_extents` and `output_extents`
- **`layer` Concept**: Requires `real_type` member type and `input_extents`, `output_extents` static data members (both `std::extents` specializations)
- **`sequential_network` Concept**: A `std::tuple` containing only connectable `layer` types
- **`is_connectable`**: Validates adjacent layers/builders have matching input/output extents
- **Input/Output Concepts**: `input<Input, Layer>`, `output<Output, Layer>`, `batch_input<BatchInput, Layer>`, `batch_output<BatchOutput, Layer>` validate type compatibility with layer shapes and real types
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

- **Module Declarations**:
  - `export module nou.core;` - Core types and network construction
  - `export module nou.predict;` - Inference and error handling (imports `nou.core`)
  - `export module nou;` - Umbrella module (imports `nou.predict`)
- **Module Re-exports**: Use `export import` to transitively expose submodule APIs
- **Internal Helpers**: Use unnamed namespace, not `export`
- **Template Specializations**: Stay internal unless explicitly exported
- **Exported APIs by Module**:
  - `nou.core`: `layer`, `sequential_network`, `is_connectable`, `build` CPO, `make_network()`, `first_layer_t`, `last_layer_t`
  - `nou.predict`: `computation_error`, `computation_errorable`, `input`, `output`, `batch_input`, `batch_output`, `propagate` CPO, `predict()` (plus re-exported `nou.core` APIs)
  - `nou`: All APIs via re-export of `nou.predict`

### Builder Implementation Options

All three patterns are tested and supported:
```cpp
// 1. Member function
struct MyBuilder {
  template <class RealType, class IndexType, std::size_t... Extents>
  auto build(std::extents<IndexType, Extents...> input_shape) const -> layer;
};

// 2. Free function (ADL via nou namespace)
template <class RealType, class IndexType, std::size_t... Extents>
auto build(const MyBuilder& b, std::extents<IndexType, Extents...> input_shape) -> layer;

// 3. Hidden friend
struct MyBuilder {
  template <class RealType, class IndexType, std::size_t... Extents>
  friend auto build(const MyBuilder& b, std::extents<IndexType, Extents...> input_shape) -> layer;
};
```

### Test Pattern

- Use `compile_fail_*` concepts to verify rejection of invalid types
- Use `static_assert` for compile-time verification (no runtime assertions)
- Layer test structs always include `.value` field for equality testing
- Tests use standard C++ blocks with comments for test names (e.g., `// --- test name --- \n { ... }`)
- For long test names (>80 characters), use multi-line comments with continuation on the next line, indented to align with the start of the test name

## Integration Points & Dependencies

### External Dependencies

- **No External Libraries**: All core functionality is implemented from scratch using only the C++23 standard library
- **Standard Library**: C++23 standard library features; `std::expected` required

### CMake Configuration

- **Language**: C++23 required, enforced by CMakeLists.txt
- **Compiler Warnings**: All warnings as errors (`-Werror` on GCC/Clang, `/WX` on MSVC)
- **Clang-Tidy**: Enabled in CI for clang++-19; specific check categories enabled (bugprone, cert, cppcoreguidelines, clang-analyzer, google, misc, modernize, performance, readability) with warnings-as-errors
- **Modules**: Uses CMake `FILE_SET CXX_MODULES` with modules listed in dependency order: `core.cppm`, `predict.cppm`, `nou.cppm`
- **Known clang-tidy false positive**: `misc-use-internal-linkage` on exported functions in the module; suppressed with `// NOLINTNEXTLINE(misc-use-internal-linkage)`

### CI/CD

- **Linux CI** (.github/workflows/linux.yml): Tests clang++-19
- **Linting**: clang++-19 job runs clang-tidy-19

## Critical Developer Knowledge

### Shape Matching Validation

The `is_connectable` trait recursively validates that:
- First argument is a `layer`
- Each subsequent argument is either: a `layer` with matching `input_extents`, OR a builder that can produce one
- Output extents of layer N (`output_extents`) must equal input extents of layer N+1 (`input_extents`)

### Input/Output Type Validation

The `input`, `output`, `batch_input`, and `batch_output` concepts use the internal `is_extents_match` trait to validate:
- Single-sample inputs/outputs: container element type matches `Layer::real_type`, and shape matches layer's extents (rank 1 for single-sample)
- Batch inputs/outputs: range of single-sample inputs/outputs (rank 2: `[batch_size, features]`)
- Error-free vs fallible types: handles both plain `T` and `std::expected<T, computation_error>`
- Scalar inputs: for single-element extents (`std::extents<std::size_t, 1>`), accepts plain `real_type`
- Multi-dimensional support: recursively validates nested ranges for higher-rank extents

### Why Compile-Time?

All computations are `constexpr` to ensure:
- Zero runtime overhead
- Type safety at compile time
- Early error detection during compilation
- Conditional error handling at zero cost when layers return plain `T` (no `std::expected` path taken)

### Adding New Tests

1. Create [test/unit_test/new_test.cc](../test/unit_test/)
2. Import the appropriate module(s):
   - `import nou;` - For tests using the full API (most common)
   - `import nou.core;` - For tests focusing only on network construction
   - `import nou.predict;` - For tests requiring both core and prediction APIs
3. Add `nou_test(new_test)` to [test/unit_test/CMakeLists.txt](../test/unit_test/CMakeLists.txt)
4. Use `static_assert` for compile-time checks and `compile_fail_*` concepts for rejection validation
5. Use standard C++ blocks with comment headers for test names (e.g., `// --- test name --- \n { ... }`)
6. For long test names (>80 characters), break into multi-line comments with continuation indented to align with the test name start

## Format & Style

- **Clang-format**: [.clang-format](../.clang-format) enforces 80-column limit, 2-space indent
- **Clang-tidy**: [.clang-tidy](../.clang-tidy) enforces modern C++, no NOLINT comments without reason
- **Comment Style**: Detailed Doxygen comments on exported types and functions. Must follow the wording style in C++ standard library documentation.
