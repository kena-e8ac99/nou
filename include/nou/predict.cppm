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
 * This enumeration represents error conditions that can occur during neural
 * network computations, including layer propagation, training, and evaluation.
 * The enum is designed to grow incrementally as new computational operations
 * are implemented and their error cases are discovered.
 *
 * The underlying type is `std::uint8_t` (1 byte) to minimize memory overhead
 * in `std::expected` and cache utilization. This provides 256 possible error
 * values, which is more than sufficient for neural network computation errors.
 *
 * @see predict for how error types are composed across network layers.
 * @see propagate for the underlying computation that may produce errors.
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
 * @brief Specifies that a type can represent either a valid computation result
 * or a computation error.
 *
 * The concept computation_errorable<T> specifies that T is a type that can
 * represent the result of a neural network computation which may fail with a
 * `computation_error`.
 *
 * ### Syntactic requirements
 * Let `U` be the type of the successful computation result.
 * computation_errorable<T> is satisfied only if T is exactly a specialization
 * of `std::expected<U, computation_error>`.
 *
 * @tparam T The type to check
 *
 * @see computation_error for the computation error conditions.
 */
export template <class T>
concept computation_errorable =
    std::same_as<T, std::expected<typename T::value_type, computation_error>>;

// Internal helper: Type trait to check if an input type matches layer extents.
// Primary template: Default to false for non-matching types.
template <class... Ts>
struct is_extents_match : public std::false_type {};

// Base case: Match with single-element scalar (extent == 1).
template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(Extent == 1 && std::same_as<std::remove_cvref_t<T>, RealType>)
struct is_extents_match<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

// Base case: Match non-span-compatible range with correct value type.
template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(!requires { std::span{std::declval<T&>()}; } &&
           std::ranges::input_range<T> &&
           std::same_as<std::remove_cvref_t<typename T::value_type>, RealType>)
struct is_extents_match<T, RealType, std::extents<IndexType, Extent>>
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
struct is_extents_match<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

// Recursive case: Match non-span-compatible range for multi-dimensional
// extents.
template <class T, class RealType, class IndexType, std::size_t Extent,
          std::size_t... Extents>
  requires(!requires { std::span{std::declval<T&>()}; } &&
           std::ranges::input_range<T>)
struct is_extents_match<T, RealType,
                        std::extents<IndexType, Extent, Extents...>>
    : public is_extents_match<
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
struct is_extents_match<T, RealType,
                        std::extents<IndexType, Extent, Extents...>>
    : public is_extents_match<
          std::ranges::range_value_t<std::remove_cvref_t<T>>, RealType,
          std::extents<IndexType, Extents...>> {};

/**
 * @brief Specifies that a type is compatible as layer input.
 *
 * The concept input<Input, Layer> specifies that Input is a valid input type
 * for the given Layer. The input must match the layer's real_type and
 * input_extents shape.
 *
 * ### Semantic requirements
 * input<Input, Layer> is modeled only if Layer models the layer concept and
 * either of the following holds:
 * - If `Layer::input_extents` is rank 1 with extent 1:
 *   - Input is same as `Layer::real_type`
 * - If `Layer::input_extents` is rank 1 with extent $N \ge 1$:
 *   - Input models `std::ranges::input_range`
 *   - `std::ranges::range_value_t<Input>` is the same as `Layer::real_type`
 *   - If Input is a contiguous range (or more specifically, `std::span` is
 *     constructible from Input) and its extent is known at compile-time (e.g.,
 *     `std::array`), its extent is N
 * - If `Layer::input_extents` is rank > 1 and outermost extent is $M \ge 1$:
 *   - Input models `std::ranges::input_range`
 *   - If Input is a contiguous range (or more specifically, `std::span` is
 *     constructible from Input) and its extent is known at compile-time (e.g.,
 *     `std::array`), its extent is M
 *   - The value type of Input recursively satisfy the same rules with respect
 *     to `Layer::input_extents` excluding the outermost extent
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
 * @tparam Input The input type to validate.
 * @tparam Layer The layer type that models the `layer` concept.
 */
export template <class Input, class Layer>
concept input =
    is_extents_match<Input, typename std::remove_cvref_t<Layer>::real_type,
                     decltype(auto(
                         std::remove_cvref_t<Layer>::input_extents))>::value;

/**
 * @brief Specifies that a type is compatible as batch layer input.
 *
 * The concept batch_input<BatchInput, Layer> specifies that BatchInput is a
 * valid batch input for the given Layer. A batch input is a range of individual
 * inputs, where each element satisfies the `input` concept for the layer.
 * This concept supports both fixed-size and dynamic-size batch inputs.
 *
 * ### Semantic requirements
 * batch_input<BatchInput, Layer> is modeled only if Layer models the layer
 * concept and the following holds:
 * - Disambiguation: BatchInput and Layer do not satisfy `input<BatchInput,
 *   Layer>` (if a type can satisfy both a single input and batch input, it is
 *   treated as a single input)
 * - BatchInput models `std::ranges::input_range`
 * - Each element in the range must satisfy input<Element, Layer>
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
 * @tparam BatchInput The batch input type to validate
 * @tparam Layer The layer type that models the layer concept
 *
 * @see input for individual input requirements
 */
export template <class BatchInput, class Layer>
concept batch_input =
    !input<BatchInput, Layer> && std::ranges::input_range<BatchInput> &&
    input<std::ranges::range_value_t<std::remove_cvref_t<BatchInput>>, Layer>;

/**
 * @brief Specifies that a type is compatible as layer output.
 *
 * The concept output<Output, Layer> specifies that Output is a valid output
 * type from the given Layer. The output may or may not satisfy
 * computation_errorable<Output> depending on layer`s error handling behavior.
 *
 * ### Semantic requirements
 * output<Output, Layer> is modeled only if Layer models the layer concept and
 * either of the following holds:
 * - If Output satisfies computation_errorable<Output>, `Output::value_type`
 *   matches the layer's real_type and output_extents shape
 * - If Output does not satisfy computation_errorable<Output>, Output itself
 *   matches the layer's real_type and output_extents shape
 *
 * (See `input` concept for detailed shape matching rules).
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
 * @tparam Output The output type to validate
 * @tparam Layer The layer type that models the `layer` concept
 *
 * @see computation_errorable for error-handling output types
 * @see input for shape matching rules
 */
export template <class Output, class Layer>
concept output =
    is_extents_match<std::conditional_t<computation_errorable<Output>,
                                        typename Output::value_type, Output>,
                     typename std::remove_cvref_t<Layer>::real_type,
                     decltype(auto(
                         std::remove_cvref_t<Layer>::output_extents))>::value;

/**
 * @brief Specifies that a type is compatible as batch layer output.
 *
 * The concept batch_output<BatchOutput, Layer> specifies that BatchOutput is a
 * valid batch output from the given Layer. The batch output may or may not
 * satisfy computation_errorable<BatchOutput> depending on layer`s error
 * handling behavior.
 *
 * ### Semantic requirements
 * batch_output<BatchOutput, Layer> is modeled only if Layer models the layer
 * concept and the following holds:
 * - If BatchOutput satisfies computation_errorable<BatchOutput>, let
 *   `BatchOutput::value_type` be R, then:
 *   - R models `std::ranges::input_range`
 *   - Each element of R satisfies output<Element, Layer>
 * - If BatchOutput does not satisfy computation_errorable<BatchOutput>, then:
 *   - BatchOutput models `std::ranges::input_range`
 *   - Each element of BatchOutput satisfies output<Element, Layer>
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
 * @tparam BatchOutput The batch output type to validate
 * @tparam Layer The layer type that models the layer concept
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
 * - The type (decayed `Layer`) models the `layer` concept
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

// Internal helper: Recursive implementation of predict.
template <std::size_t I = 0, class Network, class Input>
  requires(I < std::tuple_size_v<std::remove_cvref_t<Network>>) &&
          sequential_network<std::remove_cvref_t<Network>> &&
          std::invocable<decltype(propagate),
                         std::tuple_element_t<I, std::remove_cvref_t<Network>>,
                         Input>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto predict_impl(Network&& network, Input&& input) -> auto {
  auto intermediate =
      propagate(std::get<I>(network), std::forward<Input>(input));

  if constexpr (I + 1 == std::tuple_size_v<std::remove_cvref_t<Network>>) {
    return intermediate;
  } else if constexpr (computation_errorable<decltype(intermediate)>) {
    using next_layer_output_type = decltype(predict_impl<I + 1>(
        std::forward<Network>(network),
        std::declval<typename decltype(intermediate)::value_type>()));
    using result_type = std::expected<
        std::conditional_t<computation_errorable<next_layer_output_type>,
                           typename next_layer_output_type::value_type,
                           next_layer_output_type>,
        computation_error>;
    if (intermediate.has_value()) {
      return result_type{predict_impl<I + 1>(std::forward<Network>(network),
                                             std::move(intermediate).value())};
    }
    return result_type{std::unexpect, std::move(intermediate).error()};
  } else {
    return predict_impl<I + 1>(std::forward<Network>(network),
                               std::move(intermediate));
  }
}

/**
 * Predicts output by propagating input through a neural network.
 *
 * This function propagates input data through all layers of the network,
 * applying each layer's transformation in sequence. The function automatically
 * handles error propagation across layers:
 * - If all layers return values directly and the output layer returns type `T`,
 *   the result is of type `T`.
 * - If any layer returns `std::expected<T, computation_error>`, the result is
 *   `std::expected<T, computation_error>`
 *
 * **Error Handling:**
 * - Errors are short-circuited: if a layer returns an error, propagation to
 *   subsequent layers is skipped
 * - The first encountered error is propagated to the caller
 *
 * @tparam Network The network type (must satisfy `sequential_network` concept)
 * @tparam Input The input type (must be compatible with first layer's input)
 *
 * @param network The neural network to use for prediction
 * @param input The input data to propagate through the network
 *
 * @return Either `T` (if all layers are error-free) or
 *         `std::expected<T, computation_error>` (if any layer may fail),
 *         where `T` is the output type of the final layer
 *
 * @note This function is fully `constexpr` and can be evaluated at
 *       compile-time if all layer operations are `constexpr`.
 *
 * @see computation_error for possible error conditions
 * @see propagate for the underlying layer computation
 */
export template <class Network, input<first_layer_t<Network>> Input>
  requires sequential_network<std::remove_cvref_t<Network>> && requires {
    predict_impl(std::declval<Network>(), std::declval<Input>());
  }
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto predict(Network&& network, Input&& input) noexcept(noexcept(
    predict_impl(std::forward<Network>(network), std::forward<Input>(input))))
    -> output<output_layer_t<Network>> auto {
  return predict_impl(std::forward<Network>(network),
                      std::forward<Input>(input));
}

export template <class Network, batch_input<first_layer_t<Network>> Input>
  requires sequential_network<std::remove_cvref_t<Network>> && requires {
    predict_impl(std::declval<Network>(), std::declval<Input>());
  }
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto predict(Network&& network, Input&& input) noexcept(noexcept(
    predict_impl(std::forward<Network>(network), std::forward<Input>(input))))
    -> batch_output<output_layer_t<Network>> auto {
  return predict_impl(std::forward<Network>(network),
                      std::forward<Input>(input));
}

}  // namespace nou
