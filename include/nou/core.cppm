module;

#include <cstddef>
#include <mdspan>
#include <tuple>
#include <type_traits>
#include <utility>

export module nou.core;

namespace nou {

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
concept layer =
    requires {
      typename T::real_type;
      T::input_extents;
      T::output_extents;
    } && (T::input_extents == std::extents{T::input_extents}) &&
    (T::output_extents == std::extents{T::output_extents});

// Internal helper: Type trait to detect if a type is a sequential network
// (tuple of layers). Primary template matches non-network types.
template <class T>
struct is_sequential_network : public std::false_type {};

// Specialization matches tuples containing only layer types.
template <layer... Ts>
struct is_sequential_network<std::tuple<Ts...>> : public std::true_type {};

/**
 * Specifies that a type is a sequential neural network.
 *
 * A sequential network is a tuple-like container of `layer` types where each
 * layer's output shape matches the next layer's input shape. Layers are
 * processed in order during propagation.
 *
 * Sequential networks are composed using `make_network()` and evaluated using
 * `predict()`. This concept distinguishes sequential networks from other
 * possible network topologies (e.g., graph-based networks) that may be added
 * in the future.
 *
 * @see make_network for how sequential networks are constructed.
 * @see predict for how sequential networks are evaluated.
 */
export template <class T>
concept sequential_network = is_sequential_network<T>::value;

export template <class Network>
  requires sequential_network<std::remove_cvref_t<Network>>
using first_layer_t = std::tuple_element_t<0, std::remove_cvref_t<Network>>;

export template <class Network>
  requires sequential_network<std::remove_cvref_t<Network>>
using output_layer_t =
    std::tuple_element_t<std::tuple_size_v<std::remove_cvref_t<Network>> - 1,
                         std::remove_cvref_t<Network>>;

struct build_impl {
  template <class RealType, class T, class IndexType, std::size_t... Extents>
  constexpr auto operator()(T&& builder,
                            std::extents<IndexType, Extents...> extents) const
      noexcept(
          noexcept(std::forward<T>(builder).template build<RealType>(extents)))
          -> layer auto
    requires requires {
      { std::forward<T>(builder).template build<RealType>(extents) } -> layer;
    }
  {
    return std::forward<T>(builder).template build<RealType>(
        std::move(extents));
  }

  template <class RealType, class T, class IndexType, std::size_t... Extents>
  constexpr auto operator()(T&& builder,
                            std::extents<IndexType, Extents...> extents) const
      noexcept(noexcept(build<RealType>(std::forward<T>(builder), extents)))
          -> layer auto
    requires(
        !requires {
          std::forward<T>(builder).template build<RealType>(extents);
        } &&
        requires {
          { build<RealType>(std::forward<T>(builder), extents) } -> layer;
        })
  {
    return build<RealType>(std::forward<T>(builder), std::move(extents));
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
export inline constexpr auto build = build_impl{};
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
  requires(First::output_extents == Second::input_extents)
struct is_connectable<First, Second, Rest...>
    : public is_connectable<Second, Rest...> {};

// Recursive case: A layer followed by a builder can connect if the builder
// can be instantiated with the layer's real_type and output_shape_type.
// The instantiated layer replaces the builder for further validation.
template <layer First, class Second, class... Rest>
  requires requires {
    {
      build.template operator()<typename First::real_type>(
          std::declval<Second>(), First::output_extents)
    } -> layer;
  }
struct is_connectable<First, Second, Rest...>
    : public is_connectable<
          First,
          decltype(build.template operator()<typename First::real_type>(
              std::declval<Second>(), First::output_extents)),
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
 * @return A network object satisfying the `sequential_network` concept.
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
constexpr auto make_network(First first, Second second, Rest... rest)
    -> sequential_network auto {
  constexpr auto builder = []<std::size_t I, class T>(T& tuple) -> layer auto {
    if constexpr (layer<std::tuple_element_t<I, T>>) {
      return std::move(std::get<I>(tuple));
    } else {
      using prev_layer_type = std::tuple_element_t<I - 1, T>;
      return build.template operator()<typename prev_layer_type::real_type>(
          std::move(std::get<I>(tuple)), prev_layer_type::output_extents);
    }
  };
  return [&]<std::size_t... Is>(auto tuple, std::index_sequence<Is...>)
             -> sequential_network auto {
    return std::tuple{[&]() -> layer auto {
      return builder.template operator()<Is>(tuple);
    }()...};
  }(std::tuple{std::move(first), std::move(second), std::move(rest)...},
             std::make_index_sequence<sizeof...(Rest) + 2>{});
}

}  // namespace nou
