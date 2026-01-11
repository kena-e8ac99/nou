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
 * Unified error type for neural network computation failures.
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
   * - Value range violation (if enforced by a layer)
   */
  invalid_input,

  /**
   * Numerical instability detected during computation.
   *
   * Common causes:
   * - Overflow in floating-point arithmetic
   * - Underflow causing loss of precision
   * - Accumulation of rounding errors
   */
  numerical_instability
};

/**
 * Specifies that a type is a computation result that may contain an error.
 *
 * This concept identifies a type that represents the result of a neural
 * network computation which may fail with a `computation_error`. It is
 * specifically designed to detect `std::expected<T, computation_error>`
 * specializations.
 *
 * @tparam T The type to check
 *
 * @see computation_error for the unified error type.
 * @see predict for how computation results are composed across layers.
 */
export template <class T>
concept computation_errorable =
    std::same_as<T, std::expected<typename T::value_type, computation_error>>;

template <class... Ts>
struct is_extents_match : public std::false_type {};

template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(Extent == 1 && std::same_as<std::remove_cvref_t<T>, RealType>)
struct is_extents_match<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(!requires { std::span{std::declval<T&>()}; } &&
           std::ranges::input_range<T> &&
           std::same_as<std::remove_cvref_t<typename T::value_type>, RealType>)
struct is_extents_match<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

template <class T, class RealType, class IndexType, std::size_t Extent>
  requires(
      requires { std::span{std::declval<T&>()}; } &&
      std::same_as<std::ranges::range_value_t<std::remove_cvref_t<T>>,
                   RealType> &&
      (decltype(std::span{std::declval<T&>()})::extent == Extent ||
       decltype(std::span{std::declval<T&>()})::extent == std::dynamic_extent))
struct is_extents_match<T, RealType, std::extents<IndexType, Extent>>
    : public std::true_type {};

template <class T, class RealType, class IndexType, std::size_t Extent,
          std::size_t... Extents>
  requires(!requires { std::span{std::declval<T&>()}; } &&
           std::ranges::input_range<T>)
struct is_extents_match<T, RealType,
                        std::extents<IndexType, Extent, Extents...>>
    : public is_extents_match<
          std::ranges::range_value_t<std::remove_cvref_t<T>>, RealType,
          std::extents<IndexType, Extents...>> {};

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

export template <class Input, class Layer>
concept input =
    is_extents_match<Input, typename std::remove_cvref_t<Layer>::real_type,
                     decltype(auto(
                         std::remove_cvref_t<Layer>::input_extents))>::value;

export template <class Output, class Layer>
concept output =
    is_extents_match<std::conditional_t<computation_errorable<Output>,
                                        typename Output::value_type, Output>,
                     typename std::remove_cvref_t<Layer>::real_type,
                     decltype(auto(
                         std::remove_cvref_t<Layer>::output_extents))>::value;

export template <class BatchInput, class Layer>
concept batch_input =
    !input<BatchInput, Layer> && std::ranges::input_range<BatchInput> &&
    input<std::ranges::range_value_t<std::remove_cvref_t<BatchInput>>, Layer>;

export template <class BatchOutput, class Layer>
concept batch_output =
    (computation_errorable<BatchOutput> &&
     std::ranges::input_range<typename BatchOutput::value_type> &&
     output<std::ranges::range_value_t<typename BatchOutput::value_type>,
            Layer>) ||
    (!computation_errorable<BatchOutput> &&
     std::ranges::input_range<BatchOutput> &&
     output<std::ranges::range_value_t<BatchOutput>, Layer>);

struct propagate_impl {
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
 * Customization point object for invoking layer forward propagation.
 *
 * The `propagate` customization point object provides a uniform interface for
 * invoking layer forward propagation in neural network computations. It
 * supports multiple customization patterns, allowing users to define layer
 * propagation with either member functions or free/hidden friend functions.
 *
 * **Invocation Patterns:**
 * 1. **Member function**: If the layer type `Layer` provides a member function
 *    `propagate(input)`, it will be invoked as `layer.propagate(input)`.
 * 2. **Free or hidden friend function**: If the layer type `Layer` does not
 *    provide a member function but has a free function or hidden friend
 *    function `propagate(layer, input)`, it will be invoked via ADL
 *    (Argument Dependent Lookup).
 *
 * The customization point object is instantiated as the inline variable
 * `propagate` in the inline `cpo` namespace, enabling convenient usage
 * in network computations.
 *
 * **Return Types:**
 * - If the layer is error-free (cannot produce errors during computation),
 *   `propagate` returns `T` directly, where `T` is the output type of the
 *   layer.
 * - If the layer may produce errors during computation, `propagate` returns
 *   `std::expected<T, computation_error>`, enabling error propagation through
 *   the network.
 *
 * **Error Handling:**
 * When a layer returns `std::expected<T, computation_error>`, the `predict()`
 * function automatically handles error propagation:
 * - Errors are short-circuited: if a layer returns an error, subsequent layers
 *   are not executed
 * - The first encountered error is propagated to the caller
 * - If all layers are error-free, the final result is `T` (zero overhead)
 *
 * @tparam Layer The layer type (must satisfy `layer` concept)
 * @tparam Input The input type (must be compatible with the layer's
 *               `input_shape_type`)
 *
 * @return Either `T` (error-free computation) or
 *         `std::expected<T, computation_error>` (fallible computation),
 *         where `T` is the output type produced by the layer's propagation
 *
 * @note This function is fully `constexpr` and can be evaluated at
 *       compile-time if the layer's propagation operation is `constexpr`.
 *
 * @see predict for how propagation results are composed across network layers.
 * @see computation_error for possible error conditions.
 * @see computation_errorable for identifying layers that may produce errors.
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
