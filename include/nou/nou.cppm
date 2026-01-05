module;

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <tuple>
#include <type_traits>
#include <utility>

export module nou;

namespace nou {

/**
 * A compile-time shape descriptor template for neural network dimensions.
 *
 * This template struct represents a single dimension size in a neural network
 * shape, constrained to positive integral values known at compile-time.
 *
 * The struct exposes the dimension value through:
 * - `type`  : the type of the dimension constant (e.g., int)
 * - `value` : the dimension size as a `constexpr` value
 *
 * @tparam N A positive integral constant representing the dimension size.
 */
export template <std::integral auto N>
  requires(N > 0)
struct shape {
  using type = decltype(N);
  static constexpr type value = N;
};

/**
 * Specifies that a type can be used as a neural network layer.
 *
 * The layer concept specifies that a type provides the following member types
 * required for neural network layers.
 * - `real_type`: a type used for computations (typically floating-point, but
 * can be others)
 * - `input_shape_type`: a specialization of `shape`, representing the input
 *   dimension of the layer
 * - `output_shape_type`: a specialization of `shape`, representing the output
 * dimension of the layer
 */
export template <class T>
concept layer = requires {
  typename T::real_type;
  typename T::input_shape_type;
  typename T::output_shape_type;
  requires std::same_as<typename T::input_shape_type,
                        shape<T::input_shape_type::value>>;
  requires std::same_as<typename T::output_shape_type,
                        shape<T::output_shape_type::value>>;
};

// Internal helper: Type trait to detect if a type is a network (tuple of
// layers). Primary template matches non-network types.
template <class T>
struct is_network : public std::false_type {};

// Specialization matches tuples containing only layer types.
template <layer... Ts>
struct is_network<std::tuple<Ts...>> : public std::true_type {};

/**
 * Specifies that a type can be used as a neural network.
 *
 * The network concept specifies that a type is a tuple-like container of
 * `layers`.
 */
export template <class T>
concept network = is_network<T>::value;

struct build_impl {
  template <class RealType, class Shape, class T>
    requires requires {
      { std::declval<T>().template build<RealType, Shape>() } -> layer;
    }
  constexpr auto operator()(T&& builder) const noexcept(
      noexcept(std::forward<T>(builder).template build<RealType, Shape>()))
      -> layer auto {
    return std::forward<T>(builder).template build<RealType, Shape>();
  }

  template <class RealType, class Shape, class T>
    requires(
        !requires { std::declval<T>().template build<RealType, Shape>(); } &&
        requires {
          { build<RealType, Shape>(std::declval<T>()) } -> layer;
        })
  constexpr auto operator()(T&& builder) const
      noexcept(noexcept(build<RealType, Shape>(std::forward<T>(builder))))
          -> layer auto {
    return build<RealType, Shape>(std::forward<T>(builder));
  }
};
inline namespace cpo {
/**
 * Customization point object for invoking layer builders.
 *
 * The `build` customization point object provides a uniform interface for
 * invoking layer builders in neural network construction. It supports multiple
 * customization patterns, allowing users to define layer builders with either
 * member functions or free/hidden friend functions.
 *
 * **Invocation Patterns:**
 * 1. **Member function**: If the builder type `T` provides a member function
 *    template `build<RealType, Shape>()`, it will be invoked as
 *    `builder.build<RealType, Shape>()`.
 * 2. **Free or hidden friend function**: If the builder type `T` does not
 *    provide a member function but has a free function or hidden friend
 *    function template `build<RealType, Shape>(builder)`, it will be invoked
 *    via ADL (Argument Dependent Lookup).
 *
 * The customization point object is instantiated as the inline variable `build`
 * in the inline `cpo` namespace, enabling convenient usage in network
 * construction.
 *
 * @tparam RealType The type used for computations (typically floating-point,
 *                  but can be others)
 * @tparam Shape A specialization of `shape` representing the input dimension
 *               for the layer to be built.
 * @tparam T The type of the layer builder object.
 *
 * @return A layer satisfying the `layer` concept.
 *
 * @note The returned layer's `input_shape_type` will typically match the
 *       provided `Shape` template parameter, and its `output_shape_type` is
 *       determined by the builder's implementation.
 *
 * @see make_network for how this CPO is used in network construction.
 */
inline constexpr auto build = build_impl{};
}  // namespace cpo

// Internal helper: Type trait to check if a sequence of layers/builders can be
// connected to form a valid network. A builder is a type that can be converted
// to a layer via the build CPO. Validates shape compatibility between
// consecutive layers and ensures builders can be instantiated.
// Primary template: No types or incompatible types cannot be connected.
template <class... Ts>
struct is_connectable : public std::false_type {};

// Base case: A single layer is trivially connectable.
template <layer Layer>
struct is_connectable<Layer> : public std::true_type {};

// Recursive case: Two consecutive layers can connect if their output/input
// shapes match. Recursively checks the remaining layers.
template <layer First, layer Second, class... Rest>
  requires(
      static_cast<std::common_type_t<typename First::output_shape_type::type,
                                     typename Second::input_shape_type::type>>(
          First::output_shape_type::value) ==
      static_cast<std::common_type_t<typename First::output_shape_type::type,
                                     typename Second::input_shape_type::type>>(
          Second::input_shape_type::value))
struct is_connectable<First, Second, Rest...>
    : public is_connectable<Second, Rest...> {};

// Recursive case: A layer followed by a builder can connect if the builder
// can be instantiated with the layer's real_type and output_shape_type.
// The instantiated layer replaces the builder for further validation.
template <layer First, class Second, class... Rest>
  requires requires {
    {
      build.template
      operator()<typename First::real_type, typename First::output_shape_type>(
          std::declval<Second>())
    } -> layer;
  }
struct is_connectable<First, Second, Rest...>
    : public is_connectable<
          First,
          decltype(build.template operator()<typename First::real_type,
                                             typename First::output_shape_type>(
              std::declval<Second>())),
          Rest...> {};

/**
 * Constructs a neural network by composing layers or layer builders.
 *
 * This function constructs a neural network from a sequence of layers or layer
 * builders. Each argument is processed in order:
 * - If the argument is a layer, it is copied or moved directly into the
 * network.
 * - If the argument is a layer builder, it is instantiated by invoking the
 * `build` customization point object with the previous layer's `real_type` and
 * `output_shape_type` (see @pre).
 *
 * @tparam First The type of the first layer.
 * @tparam Second The type of the second layer or layer builder.
 * @tparam Rest The types of the remaining layers or layer builders.
 *
 * @param first The first layer in the network.
 * @param second The layer or layer builder for the second layer in the network.
 * @param rest The layers or layer builders for the remaining layers in the
 * network.
 *
 * @return A network object satisfying the `network` concept.
 *
 * @pre For each adjacent pair of arguments `(Prev, Next)`:
 * - `Prev` must satisfy the `layer` concept
 * - `Next` must satisfy one of the following:
 *   - **`Next` satisfies the `layer` concept**:
 *     `Prev::output_shape_type` must match `Next::input_shape_type`.
 *   - **`Next` is a layer builder**:
 *     `build` customization point object must be invocable and return a type
 *     satisfying the `layer` concept. The resulting layer becomes the next
 *     `Prev` in the sequence.
 *
 * @see build for details on how layer builders are invoked.
 */
export template <layer First, class Second, class... Rest>
  requires(is_connectable<First, Second, Rest...>::value)
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto make_network(First first, Second second, Rest... rest) -> network
    auto {
  constexpr auto builder = []<std::size_t I, class T>(T& tuple) -> layer auto {
    if constexpr (layer<std::tuple_element_t<I, T>>) {
      return std::move(std::get<I>(tuple));
    } else {
      using prev_layer_type = std::tuple_element_t<I - 1, T>;
      using real_type = prev_layer_type::real_type;
      using input_shape_type = prev_layer_type::output_shape_type;

      return build.template operator()<real_type, input_shape_type>(
          std::move(std::get<I>(tuple)));
    }
  };
  return [&]<std::size_t... Is>(auto tuple, std::index_sequence<Is...>)
             -> network auto {
    return std::tuple{[&]() -> layer auto {
      return builder.template operator()<Is>(tuple);
    }()...};
  }(std::tuple{std::move(first), std::move(second), std::move(rest)...},
             std::make_index_sequence<sizeof...(Rest) + 2>{});
}

struct propagate_impl {
  template <class Layer, class Input>
    requires layer<std::remove_cvref_t<Layer>> && requires {
      { std::declval<Layer>().propagate(std::declval<Input>()) };
    }
  constexpr auto operator()(Layer&& layer, Input&& input) const
      noexcept(noexcept(
          std::forward<Layer>(layer).propagate(std::forward<Input>(input))))
          -> decltype(std::forward<Layer>(layer).propagate(
              std::forward<Input>(input))) {
    return std::forward<Layer>(layer).propagate(std::forward<Input>(input));
  }

  template <class Layer, class Input>
    requires(
        layer<std::remove_cvref_t<Layer>> &&
        !requires { std::declval<Layer>().propagate(std::declval<Input>()); } &&
        requires {
          { propagate(std::declval<Layer>(), std::declval<Input>()) };
        })
  constexpr auto operator()(Layer&& layer, Input&& input) const
      noexcept(noexcept(propagate(std::forward<Layer>(layer),
                                  std::forward<Input>(input))))
          -> decltype(propagate(std::forward<Layer>(layer),
                                std::forward<Input>(input))) {
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
 * **Design Philosophy:**
 * - Layers that cannot produce errors return values directly (type `T`)
 * - Layers that may produce errors return `std::expected<T, computation_error>`
 * - Operations like `predict()`, `fit()`, `evaluate()` propagate this unified
 *   error type across all layers
 * - This conditional approach achieves zero overhead for error-free paths
 * - Single error type enables consistent error handling across all operations
 *
 * **Adding new values:**
 * - New values are added incrementally during operation/layer implementation
 * - Add a value when its error condition is discovered in actual code
 * - Keep values focused on actionable runtime errors
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
 * This concept identifies a types that represents the result of a neural
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

// Internal helper: Recursive implementation of predict.
template <std::size_t I, class Network, class Input>
  requires(I < std::tuple_size_v<std::remove_cvref_t<Network>>) &&
          network<std::remove_cvref_t<Network>> &&
          std::invocable<decltype(propagate),
                         std::tuple_element_t<I, std::remove_cvref_t<Network>>,
                         Input>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto predict_impl(Network&& network, Input&& input) noexcept(noexcept(
    propagate(std::get<I>(network), std::forward<Input>(input)))) -> auto {
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
 * @tparam Network The network type (must satisfy `network` concept)
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
export template <class Network, class Input>
  requires network<std::remove_cvref_t<Network>> &&
           std::invocable<decltype(propagate),
                          std::tuple_element_t<0, std::remove_cvref_t<Network>>,
                          Input>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto predict(Network&& network, Input&& input) noexcept(
    noexcept(predict_impl<0>(std::forward<Network>(network),
                             std::forward<Input>(input)))) -> auto {
  return predict_impl<0>(std::forward<Network>(network),
                         std::forward<Input>(input));
}

}  // namespace nou
