module;

#include <concepts>
#include <cstddef>
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

// Internal helper: Type trait to detect if a type is a network (tuple of layers).
// Primary template matches non-network types.
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
inline constexpr auto build = build_impl{};
}

// Internal helper: Type trait to check if a sequence of layers/builders can be
// connected to form a valid network. Validates shape compatibility between
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
 *     `build<Prev::real_type, Prev::output_shape_type>(Next)` customization
 *     point object must be invocable and return a type satisfying the `layer`
 *     concept.
 *     The resulting layer becomes the next `Prev` in the sequence.
 */
export template <layer First, class Second, class... Rest>
  requires(is_connectable<First, Second, Rest...>::value)
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

}  // namespace nou
