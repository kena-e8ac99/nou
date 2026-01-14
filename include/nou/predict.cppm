module;

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <mdspan>
#include <ranges>
#include <span>
#include <tuple>
#include <type_traits>
#include <utility>

export module nou.predict;

export import nou.core;

namespace nou {

/**
 * @brief Unified error type for neural network computation failures.
 *
 * The enumeration `computation_error` represents error conditions that can
 * occur during neural network computations, including layer propagation,
 * training, and evaluation. The enum is designed to grow incrementally as new
 * computational operations are implemented and their error cases are
 * discovered.
 *
 * The underlying type is `std::uint8_t` (1 byte) to minimize memory overhead
 * in `std::expected` and cache utilization. This provides 256 possible error
 * values, which is more than sufficient for neural network computation errors.
 */
export enum class computation_error : std::uint8_t {
  /**
   * Input values are outside the expected domain.
   *
   * Common causes:
   * - NaN (Not a Number) detected in input or intermediate computation
   * - Inf (Infinity) detected in input or intermediate computation
   * - Value range violation (e.g., probability input not in [0, 1])
   */
  invalid_input,

  /**
   * Numerical instability detected during computation.
   *
   * Common causes:
   * - Overflow in floating-point arithmetic
   * - Underflow causing loss of precision
   * - Accumulation of rounding errors leading to invalid results
   */
  numerical_instability
};

/**
 * @brief Specifies that a type represents either a valid computation result
 * or a computation error.
 *
 * The concept `computation_errorable<T>` specifies that `T` is a type that
 * represents the result of a neural network computation which may fail with a
 * `computation_error`.
 *
 * ### Syntactic requirements
 * Let `U` be the type of the successful computation result.
 * `computation_errorable<T>` is satisfied only if `T` is exactly a
 * specialization of `std::expected<U, computation_error>`.
 *
 * @tparam T The type to check
 *
 * @see computation_error for the computation error conditions.
 */
export template <class T>
concept computation_errorable = requires {
  typename T::value_type;
} && std::same_as<T, std::expected<typename T::value_type, computation_error>>;

// Internal helper: Type trait to check if an input type matches layer extents.
// Primary template: Default to false for non-matching types.
template <class... Ts>
struct is_compatible_extents_with : public std::false_type {};

// Base case: Match with single-element scalar (extent == 1).
template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(Extent == 1 && std::same_as<std::remove_cvref_t<T>, RealType>)
struct is_compatible_extents_with<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

// Base case: Match non-span-compatible range with correct value type.
template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(!requires { std::span{std::declval<T&>()}; } &&
           std::ranges::input_range<T> &&
           std::same_as<std::remove_cvref_t<typename T::value_type>, RealType>)
struct is_compatible_extents_with<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

// Base case: Match span-compatible range with correct extent or dynamic extent
// and correct value type.
template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(
      requires { std::span{std::declval<T&>()}; } &&
      std::same_as<std::ranges::range_value_t<std::remove_cvref_t<T>>,
                   RealType> &&
      (decltype(std::span{std::declval<T&>()})::extent == Extent ||
       decltype(std::span{std::declval<T&>()})::extent == std::dynamic_extent))
struct is_compatible_extents_with<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

// Recursive case: Match non-span-compatible range for multi-dimensional
// extents.
template <class T, class RealType, class IndexType, std::size_t Extent,
          std::size_t... Extents>
  requires(!requires { std::span{std::declval<T&>()}; } &&
           std::ranges::input_range<T>)
struct is_compatible_extents_with<T, RealType,
                                  std::extents<IndexType, Extent, Extents...>>
    : public is_compatible_extents_with<
          std::ranges::range_value_t<std::remove_cvref_t<T>>, RealType,
          std::extents<IndexType, Extents...>> {};

// Recursive case: Match span-compatible range for multi-dimensional extents
// with correct extent or dynamic extent.
template <class T, class RealType, class IndexType, std::size_t Extent,
          std::size_t... Extents>
  requires(requires {
    std::span{std::declval<T&>()};
  } && (decltype(std::span{std::declval<T&>()})::extent == Extent ||
        decltype(std::span{std::declval<T&>()})::extent == std::dynamic_extent))
struct is_compatible_extents_with<T, RealType,
                                  std::extents<IndexType, Extent, Extents...>>
    : public is_compatible_extents_with<
          std::ranges::range_value_t<std::remove_cvref_t<T>>, RealType,
          std::extents<IndexType, Extents...>> {};

/**
 * @brief Specifies that a type is compatible with given extents and real type.
 *
 * The concept `compatible_extents_with<T, RealType, Extents>` specifies that
 * `T` can be used as a container or value matching the shape described by
 * `Extents` and the element type `RealType`.
 *
 * ### Semantic requirements
 * `compatible_extents_with<T, RealType, Extents>` is modeled only if
 * - If `Extents` is rank 1 with extent 1:
 *   - `T` is same as `RealType`
 * - If `Extents` is rank 1 with extent \f$N \ge 1\f$:
 *   - `T` models `std::ranges::input_range`
 *   - `std::ranges::range_value_t<T>` is the same as `RealType`
 *   - If `T` is a contiguous range (or more specifically, `std::span` is
 *     constructible from `T`) and its extent is known at compile-time
 *     (e.g., `std::array`), its extent is `N`
 * - If `Extents` is rank > 1:
 *  - `T` models `std::ranges::input_range`
 *  - If `T` is a contiguous range (or more specifically, `std::span` is
 *    constructible from `T`) and its extent is known at compile-time
 *    (e.g., `std::array`), its extent is equal to the outermost extent of
 *    `Extents`
 *  - The value type of `T` recursively models the same rules with respect to
 *    `Extents` excluding the outermost extent
 *
 * ### Example
 * ```cpp
 * static_assert(nou::compatible_extents_with<double, double,
 *               std::extents<std::size_t, 1>>); // scalar
 * static_assert(nou::compatible_extents_with<std::array<float, 3>, float,
 *               std::extents<std::size_t, 3>>); // fixed-size 1D
 * static_assert(nou::compatible_extents_with<std::vector<float>, float,
 *               std::extents<std::size_t, 3>>); // dynamic-size 1D
 * static_assert(nou::compatible_extents_with<std::array<std::array<int, 2>, 4>,
 *               int, std::extents<std::size_t, 4, 2>>); // fixed-size 2D
 * static_assert(nou::compatible_extents_with<std::vector<std::array<int, 2>>,
 *               int, std::extents<std::size_t, 4, 2>>); // dynamic-size 2D
 * ```
 *
 * @tparam T The type to be checked.
 * @tparam RealType The element type to be matched.
 * @tparam Extents The extents type describing the shape to be matched.
 */
export template <class T, class RealType, class Extents>
concept compatible_extents_with =
    is_compatible_extents_with<T, RealType, Extents>::value;

/**
 * @brief Specifies that a type is compatible as layer single input.
 *
 * The concept `input<Input, Layer>` specifies that `Input` is a valid input
 * type for the given `Layer`. The input must match the layer's `real_type` and
 * `input_extents` shape.
 *
 * ### Semantic requirements
 * Let `L` be the type of `layer` after removing cv-ref qualifiers.
 * `input<Input, Layer>` is modeled only if `Layer` models the `layer` and
 * `compatible_extents_with<Input, L::real_type, L::input_extents>` is modeled.
 *
 * (See `compatible_extents_with` for detailed shape matching rules).
 *
 * ### Example
 * ```cpp
 * struct my_layer {
 *   using real_type = float;
 *   static constexpr std::extents<std::size_t, 3> input_extents{};
 *   // ...
 * };
 * static_assert(nou::input<std::array<float, 3>, my_layer>); // fixed extent 3
 * static_assert(nou::input<std::vector<float>, my_layer>);   // dynamic extent
 * ```
 *
 * @tparam Input The input type to be checked.
 * @tparam Layer The layer type to be checked.
 *
 * @see layer for the requirements on the layer type.
 * @see compatible_extents_with for shape matching rules.
 */
export template <class Input, class Layer>
concept input = compatible_extents_with<
    Input, typename std::remove_cvref_t<Layer>::real_type,
    decltype(auto(std::remove_cvref_t<Layer>::input_extents))>;

/**
 * @brief Specifies that a type is compatible as batch layer input.
 *
 * The concept `batch_input<BatchInput, Layer>` specifies that `BatchInput` is a
 * valid batch input for the given `Layer`. A batch input is a range of
 * individual inputs, where each element models `input` for the layer. This
 * concept supports both fixed-size and dynamic-size batch inputs.
 *
 * ### Semantic requirements
 * `batch_input<BatchInput, Layer>` is modeled only if `Layer` models `layer`
 * and the following holds:
 * - Disambiguation: `BatchInput` and `Layer` do not satisfy `input<BatchInput,
 *   Layer>` (if a type can satisfy both a single input and batch input, it is
 *   treated as a single input)
 * - `BatchInput` models `std::ranges::input_range`
 * - Let `E` be `std::ranges::range_value_t<std::remove_cvref_t<BatchInput>>`,
 *   `input<E, Layer>` is modeled
 *
 * ### Example
 * ```cpp
 * struct my_layer {
 *   using real_type = float;
 *   static constexpr std::extents<std::size_t, 3> input_extents{};
 *   // ...
 * };
 * // Fixed-size batch of 10 samples
 * static_assert(
 *    nou::batch_input<std::array<std::array<float, 3>, 10>, my_layer>);
 * // Dynamic-size batch of samples
 * static_assert(
 *    nou::batch_input<std::vector<std::array<float, 3>>, my_layer>);
 * ```
 *
 * @tparam BatchInput The batch input type to be checked.
 * @tparam Layer The layer type to be checked.
 *
 * @see layer for the requirements on the layer type
 * @see input for individual input requirements
 */
export template <class BatchInput, class Layer>
concept batch_input =
    !input<BatchInput, Layer> && std::ranges::input_range<BatchInput> &&
    input<std::ranges::range_value_t<std::remove_cvref_t<BatchInput>>, Layer>;

/**
 * @brief Specifies that a type is compatible as layer single output.
 *
 * The concept `output<Output, Layer>` specifies that `Output` is a valid output
 * type from the given `Layer`. The output may or may not satisfy
 * `computation_errorable<Output>` depending on `Layer`'s error handling
 * behavior.
 *
 * ### Semantic requirements
 * Let `L` be the type of `layer` after removing cv-ref qualifiers.
 * `output<Output, Layer>` is modeled only if `Layer` models `layer` and either
 * of the following holds:
 * - If `Output` satisfies `computation_errorable<Output>`, let
 *   `Output::value_type` be `R`, then `compatible_extents_with<R, L::real_type,
 *   L::output_extents>` is modeled
 * - If `Output` does not satisfy `computation_errorable<Output>`,
 *   `compatible_extents_with<Output, L::real_type, L::output_extents>` is
 *   modeled
 *
 * (See `compatible_extents_with` for detailed shape matching rules).
 *
 * ### Example
 * ```cpp
 * struct my_layer {
 *   using real_type = float;
 *   static constexpr std::extents<std::size_t, 3> output_extents{};
 *   // ...
 * };
 * static_assert(nou::output<std::array<float, 3>, my_layer>); // fixed extent 3
 * static_assert(nou::output<std::vector<float>, my_layer>);   // dynamic extent
 * static_assert(
 *     nou::output<std::expected<std::array<float, 3>, nou::computation_error>,
 *     my_layer>); // errorable output
 * ```
 *
 * @tparam Output The output type to be checked.
 * @tparam Layer The layer type to be checked.
 *
 * @see computation_errorable for error-handling output types
 * @see compatible_extents_with for shape matching rules
 */
export template <class Output, class Layer>
concept output = compatible_extents_with<
    std::conditional_t<computation_errorable<Output>,
                       typename Output::value_type, Output>,
    typename std::remove_cvref_t<Layer>::real_type,
    decltype(auto(std::remove_cvref_t<Layer>::output_extents))>;
/**
 * @brief Specifies that a type is compatible as batch layer output.
 *
 * The concept `batch_output<BatchOutput, Layer>` specifies that `BatchOutput`
 * is a valid batch output from the given `Layer`. The batch output may or may
 * not satisfy `computation_errorable<BatchOutput>` depending on `Layer`'s error
 * handling behavior.
 *
 * ### Semantic requirements
 * `batch_output<BatchOutput, Layer>` is modeled only if `Layer` models `layer`
 * and the following holds:
 * - If `BatchOutput` satisfies `computation_errorable<BatchOutput>`, let
 *   `BatchOutput::value_type` be `R`, then:
 *   - `R` models `std::ranges::input_range`
 *   - Let `E` be `std::ranges::range_value_t<R>`, `output<E, Layer>` is modeled
 * - If `BatchOutput` does not satisfy `computation_errorable<BatchOutput>`,
 *   then:
 *   - `BatchOutput` models `std::ranges::input_range`
 *   - Let `E` be `std::ranges::range_value_t<BatchOutput>`, `output<E, Layer>`
 *     is modeled
 *
 * ### Example
 * ```cpp
 * struct my_layer {
 *   using real_type = float;
 *   static constexpr std::extents<std::size_t, 3> output_extents{};
 *   // ...
 * };
 * // Fixed-size batch of 10 samples
 * static_assert(
 *    nou::batch_output<std::array<std::array<float, 3>, 10>, my_layer>);
 * // Dynamic-size batch of samples
 * static_assert(
 *    nou::batch_output<std::vector<std::array<float, 3>>, my_layer>);
 * // Errorable batch output
 * static_assert(
 *    nou::batch_output<std::expected<std::vector<std::array<float, 3>>,
 *    nou::computation_error>, my_layer>);
 * ```
 *
 * @tparam BatchOutput The batch output type to be checked.
 * @tparam Layer The layer type to be checked.
 *
 * @see output for individual output requirements
 * @see batch_input for the corresponding input concept
 * @see computation_errorable for error-handling types
 */
export template <class BatchOutput, class Layer>
concept batch_output =
    (computation_errorable<BatchOutput> &&
     std::ranges::input_range<typename BatchOutput::value_type> &&
     output<std::ranges::range_value_t<typename BatchOutput::value_type>,
            Layer>) ||
    (!computation_errorable<BatchOutput> &&
     std::ranges::input_range<BatchOutput> &&
     output<std::ranges::range_value_t<BatchOutput>, Layer>);

// Internal helper: Implementation of the propagate customization point object.
struct propagate_impl {
  // Member function overload for single input.
  template <class Layer, input<Layer> Input>
    requires layer<std::remove_cvref_t<Layer>> && requires {
      {
        std::declval<Layer>().propagate(std::declval<Input>())
      } -> output<Layer>;
    }
  constexpr auto operator()(Layer&& layer, Input&& input) const
      noexcept(noexcept(std::forward<Layer>(layer).propagate(
          std::forward<Input>(input)))) -> output<Layer> auto {
    return std::forward<Layer>(layer).propagate(std::forward<Input>(input));
  }

  // Free function / hidden friend overload for single input.
  template <class Layer, input<Layer> Input>
    requires(
        layer<std::remove_cvref_t<Layer>> &&
        !requires { std::declval<Layer>().propagate(std::declval<Input>()); } &&
        requires {
          {
            propagate(std::declval<Layer>(), std::declval<Input>())
          } -> output<Layer>;
        })
  constexpr auto operator()(Layer&& layer, Input&& input) const
      noexcept(noexcept(propagate(std::forward<Layer>(layer),
                                  std::forward<Input>(input))))
          -> output<Layer> auto {
    return propagate(std::forward<Layer>(layer), std::forward<Input>(input));
  }

  // Member function overload for batch input.
  template <class Layer, batch_input<Layer> Input>
    requires layer<std::remove_cvref_t<Layer>> && requires {
      {
        std::declval<Layer>().propagate(std::declval<Input>())
      } -> batch_output<Layer>;
    }
  constexpr auto operator()(Layer&& layer, Input&& input) const
      noexcept(noexcept(std::forward<Layer>(layer).propagate(
          std::forward<Input>(input)))) -> batch_output<Layer> auto {
    return std::forward<Layer>(layer).propagate(std::forward<Input>(input));
  }

  // Free function / hidden friend overload for batch input.
  template <class Layer, batch_input<Layer> Input>
    requires(
        layer<std::remove_cvref_t<Layer>> &&
        !requires { std::declval<Layer>().propagate(std::declval<Input>()); } &&
        requires {
          {
            propagate(std::declval<Layer>(), std::declval<Input>())
          } -> batch_output<Layer>;
        })
  constexpr auto operator()(Layer&& layer, Input&& input) const
      noexcept(noexcept(propagate(std::forward<Layer>(layer),
                                  std::forward<Input>(input))))
          -> batch_output<Layer> auto {
    return propagate(std::forward<Layer>(layer), std::forward<Input>(input));
  }
};

inline namespace cpo {
/**
 * @brief Customization point object for propagating input through a layer.
 *
 * The name `propagate` denotes a customization point object (CPO) used to
 * perform the forward propagation of input data through a neural network layer.
 *
 * ### Call signatures
 * ```cpp
 * template <class Layer, class Input>
 *   requires /\* see below *\/
 * constexpr auto propagate(Layer&& layer, Input&& input)
 *     noexcept(noexcept(expression)) -> /\* see below *\/ auto;
 * ```
 * - `Layer`: The type of the layer to use for propagation.
 * - `Input`: The type of the input data (single or batch input) to propagate
 *   through the layer.
 *
 * ### Constraints
 * A call to `propagate` is valid only if:
 * - The type (decayed `Layer`) models `layer`
 * - The type `Input` models either the `input<Layer>` or `batch_input<Layer>`
 *
 * ### Behavior
 * Let `L` be the type of `layer` after removing cv-ref qualifiers.
 * The expression `propagate(layer, input)` is expression-equivalent to:
 * 1. `layer.propagate(input)` if that expression is valid and its result models
 *    the required output concept
 * 2. Otherwise, `propagate(layer, input)` (via ADL) if that expression is valid
 *    and its result models the required output concept
 * 3. Otherwise, the program is ill-formed
 *
 * In steps 1 and 2, the required output concept is determined as follows:
 * - If `Input` models `input<Layer>`, the result must model `output<Layer>`
 * - If `Input` models `batch_input<Layer>`, the result must model
 *   `batch_output<Layer>`
 *
 * ### Example
 * ```cpp
 * struct my_layer {
 *   using real_type = float;
 *   static constexpr std::extents<std::size_t, 3> input_extents{};
 *   static constexpr std::extents<std::size_t, 4> output_extents{};
 *
 *   // Implementation handling both single and batch input via template
 *   template <class Input>
 *   auto propagate(Input&& input) const {
 *     // In a real implementation, check if `Input` is single or batch input
 *     // and compute appropriately.
 *     return /\* ... construct output or batch output ... *\/;
 *   }
 * };
 * constexpr nou::layer my_layer layer{};
 *
 * // Single input propagation
 * constexpr nou::output<my_layer> auto output =
 *     nou::propagate(layer, std::array<float, 3>{1.0f, 2.0f, 3.0f});
 *
 * // Batch input propagation (fixed-size batch)
 * constexpr nou::batch_output<my_layer> auto batch_output =
 *     nou::propagate(layer, std::array<std::array<float, 3>, 2>{
 *         {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
 *
 * // Batch input propagation (dynamic-size batch)
 * constexpr nou::batch_output<my_layer> auto dynamic_batch_output =
 *     nou::propagate(layer, std::vector<std::array<float, 3>>{
 *         {1.0f, 2.0f, 3.0f}, {4.0f, 5.0f, 6.0f}});
 * ```
 *
 * @see layer for the requirements on `Layer` type.
 * @see input and batch_input for the requirements on `Input` type.
 * @see output and batch_output for the requirements on the result type.
 */
export inline constexpr auto propagate = propagate_impl{};
}  // namespace cpo

// Internal helper: Recursive predict implementation.
template <std::size_t, class...>
struct predict_step;

template <std::size_t I, class Network, class Input>
  requires(I == std::tuple_size_v<std::remove_cvref_t<Network>>)
struct predict_step<I, Network, Input> {
  static constexpr bool is_noexcept = true;
};

template <std::size_t I, class Network, class Input>
  requires(I < std::tuple_size_v<std::remove_cvref_t<Network>>) &&
          std::invocable<decltype(propagate),
                         std::tuple_element_t<I, std::remove_cvref_t<Network>>,
                         Input>
struct predict_step<I, Network, Input> {
  using current_layer_type =
      std::tuple_element_t<I, std::remove_cvref_t<Network>>;

  using current_output_type = decltype(propagate(
      std::declval<current_layer_type&>(), std::declval<Input>()));

  static constexpr bool is_last =
      (I + 1 == std::tuple_size_v<std::remove_cvref_t<Network>>);

  using next_input_type = std::conditional_t<
      is_last, void,
      std::conditional_t<computation_errorable<current_output_type>,
                         typename current_output_type::value_type,
                         current_output_type>>;

  static constexpr bool is_noexcept =
      noexcept(propagate(std::declval<current_layer_type&>(),
                         std::declval<Input>())) &&
      predict_step<I + 1, Network, next_input_type>::is_noexcept;

  static constexpr auto call(Network network, Input input) noexcept(is_noexcept)
      -> auto {
    auto intermediate =
        propagate(std::get<I>(network), std::forward<Input>(input));

    if constexpr (is_last) {
      return intermediate;
    } else {
      using next_step = predict_step<I + 1, Network, next_input_type>;

      if constexpr (computation_errorable<current_output_type>) {
        using next_output_type = decltype(next_step::call(
            std::declval<Network>(), std::declval<next_input_type>()));
        using result_type = std::expected<
            std::conditional_t<computation_errorable<next_output_type>,
                               typename next_output_type::value_type,
                               next_output_type>,
            computation_error>;

        if (intermediate.has_value()) {
          return result_type{next_step::call(std::forward<Network>(network),
                                             std::move(intermediate).value())};
        }
        return result_type{std::unexpect, std::move(intermediate).error()};
      } else {
        return next_step::call(std::forward<Network>(network),
                               std::move(intermediate));
      }
    }
  }
};

/**
 * @brief Predicts output from single input using the given sequential
 * network.
 *
 * This function performs forward propagation of a single input through the
 * entire sequential network, returning the final output.
 *
 * ### Error handling
 * If any layer in the network produces an errorable output (satisfying
 * `computation_errorable`), the output of `predict` will also be
 * errorable (wrapped in `std::expected`). If an intermediate layer returns a
 * `computation_error`, the process is short-circuited: the error is returned
 * immediately, and subsequent layers are not executed.
 *
 * ### Example
 * ```cpp
 * // network models nou::sequential_network
 * constexpr nou::output<last_layer_t<decltype(network)>> auto result =
 *     nou::predict(network, std::array<float, 3>{1.0f, 2.0f, 3.0f});
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model
 * `sequential_network`.
 * @tparam Input The type of the single input. This type must model
 * `input<first_layer_t<Network>>`.
 * @param network The sequential network instance.
 * @param input The single input to propagate through the network.
 * @return The final output from the last layer. The return type must model
 * `output<last_layer_t<Network>>`.
 *
 * @see sequential_network for the requirements on the Network type.
 * @see input for the requirements on the Input type.
 * @see output for the requirements on the return type.
 * @see propagate for the underlying layer propagation mechanism.
 */
export template <class Network, input<first_layer_t<Network>> Input>
  requires sequential_network<std::remove_cvref_t<Network>> && requires {
    predict_step<0, Network, Input>::call(std::declval<Network>(),
                                          std::declval<Input>());
  }
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto predict(Network&& network, Input&& input) noexcept(
    predict_step<0, Network, Input>::is_noexcept)
    -> output<last_layer_t<Network>> auto {
  return predict_step<0, Network, Input>::call(std::forward<Network>(network),
                                               std::forward<Input>(input));
}

/**
 * @brief Predicts batch output from batch input using the given sequential
 * network.
 *
 * This function performs forward propagation of multiple samples (batch
 * input) through the entire sequential network, returning the final batch
 * output.
 *
 * ### Error handling
 * Follows the same short-circuit logic as the single input version. If any
 * layer produces an error for the batch, the computation stops immediately.
 *
 * ### Example
 * ```cpp
 * // Batch input prediction (fixed or dynamic size)
 * constexpr nou::batch_output<last_layer_t<decltype(network)>> auto result =
 *     nou::predict(
 *         network, std::vector<std::array<float, 3>>{{1.0f, ...}, ...});
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model
 * `sequential_network`.
 * @tparam BatchInput The type of the batch input. This type must model
 * `batch_input<first_layer_t<Network>>`.
 * @param network The sequential network instance.
 * @param input The batch input to propagate.
 * @return The final batch output from the last layer. The return type must
 * model `batch_output<last_layer_t<Network>>`.
 *
 * @see sequential_network for the requirements on the Network type.
 * @see batch_input for the requirements on the Input type.
 * @see batch_output for the requirements on the return type.
 * @see propagate for the underlying layer propagation mechanism.
 *
 */
export template <class Network, batch_input<first_layer_t<Network>> BatchInput>
  requires sequential_network<std::remove_cvref_t<Network>> && requires {
    predict_step<0, Network, BatchInput>::call(std::declval<Network>(),
                                               std::declval<BatchInput>());
  }
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto predict(Network&& network, BatchInput&& input) noexcept(
    predict_step<0, Network, BatchInput>::is_noexcept)
    -> batch_output<last_layer_t<Network>> auto {
  return predict_step<0, Network, BatchInput>::call(
      std::forward<Network>(network), std::forward<BatchInput>(input));
}

export template <class Network, class Input>
  requires sequential_network<std::remove_cvref_t<Network>> &&
               (input<Input, first_layer_t<Network>> ||
                batch_input<Input, first_layer_t<Network>>)
using successful_predict_result_t = std::conditional_t<
    computation_errorable<decltype(predict(std::declval<Network>(),
                                           std::declval<Input>()))>,
    typename decltype(predict(std::declval<Network>(),
                              std::declval<Input>()))::value_type,
    decltype(predict(std::declval<Network>(), std::declval<Input>()))>;

/**
 * @brief Specifies that a type is a mathematical function.
 *
 * The concept `mathematical_function<T, Args...>` specifies that `T` is a
 * mathematical function that can be invoked with arguments of types `Args...`
 * and returns a non-void result.
 *
 * ### Syntactic requirements
 * Let `R` be the result type of invoking `T` with arguments of types `Args...`.
 * `mathematical_function<T, Args...>` is satisfied only if:
 * - `T` is invocable with arguments of types `Args...`
 * - `R` is not `void`
 *
 * ### Example
 * ```cpp
 * static_assert(
 *     nou::mathematical_function<double(*)(double, double), double, double>);
 * ```
 *
 * @tparam T The type to check.
 * @tparam Args The argument types for invocation.
 *
 */
export template <class T, class... Args>
concept mathematical_function =
    std::invocable<T, Args...> &&
    !std::same_as<std::invoke_result_t<T, Args...>, void>;

}  // namespace nou
