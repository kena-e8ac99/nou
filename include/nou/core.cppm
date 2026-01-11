module;

#include <concepts>
#include <cstddef>
#include <mdspan>
#include <tuple>
#include <type_traits>
#include <utility>

export module nou.core;

namespace nou {

/**
 * @brief Specifies that a type qualifies as a neural network layer.
 *
 * The concept layer<T> specifies that T is the type of a layer that can be used
 * to construct a neural network.
 *
 * ### Semantics requirements
 * layer<T> is modeled only if T provides the following members:
 * - `T::real_type`: A member type used for computations within the layer
 * - `T::input_extents` and `T::output_extents`: Static data members whose types
 *   are specializations of `std::extents`. **They describe the shape of a
 *   single sample (excluding batch dimension)**.
 *
 * ### Example
 * ```cpp
 * struct my_layer {
 *   using real_type = float;
 *   // Defines a 1D input of size 2 (per sample).
 *   // Note: batch dimension is not included here.
 *   static constexpr std::extents<std::size_t, 2> input_extents{};
 *   // Defines a 1D output of size 3 (per sample).
 *   static constexpr std::extents<std::size_t, 3> output_extents{};
 *   // ...
 * };
 * static_assert(nou::layer<my_layer>);
 * ```
 *
 * @tparam T The type to be checked.
 * @note This concept does not describe how the layer performs forward
 *       propagation, back-propagation, or parameter updates. Those behaviors
 *       are modeled separately.
 * @see make_network for how layers are composed into networks.
 * @see propagate for concepts that layers should satisfy during propagation.
 */
export template <class T>
concept layer =
    requires {
      typename T::real_type;
      T::input_extents;
      T::output_extents;
    } &&
    std::same_as<decltype(auto(T::input_extents)),
                 decltype(std::extents{T::input_extents})> &&
    std::same_as<decltype(auto(T::output_extents)),
                 decltype(std::extents{T::output_extents})>;

// Internal helper: Implementation of the build customization point object.
struct build_impl {
  // Member function overload.
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

  // Free function / hidden friend overload.
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
 * @brief Customization point object for building layer.
 *
 * The name build denotes a customization point object (CPO) used to construct a
 * layer during neural network construction.
 *
 * ### Call signature
 * ```cpp
 * template <class RealType, class T, class IndexType, std::size_t... Extents>
 *   requires /\* see below *\/
 * constexpr auto build(T&& builder,
 *                      std::extents<IndexType, Extents...> extents)
 *     noexcept(noexcept(expression)) -> layer auto;
 * ```
 * - RealType: A type used for computations, which corresponds to the
 *   `real_type` of the preceding layer in the network.
 * - extents: The shape of a single sample (excluding batch dimension). This
 *   corresponds to the `output_extents` of the preceding layer and serves as
 *   the input shape for the layer being built.
 *
 * ### Behavior
 * A call to build is expression-equivalent to:
 * 1. `builder.template build<RealType>(extents)` if that expression is
 *    valid and its result models the `layer`
 * 2. Otherwise, `build<RealType>(builder, extents)` if that expression is valid
 *    and its result models the `layer`, where the meaning of build is
 *    established as if by performing argument-dependent lookup (ADL) only
 * 3. Otherwise, the program is ill-formed.
 *
 * ### Example
 * ```cpp
 * struct my_layer_builder {
 *   static constexpr std::size_t output_features = 3;
 *
 *   template <class RealType, class IndexType, std::size_t... Extents>
 *   auto build(std::extents<IndexType, Extents...> input_shape) const {
 *     // Logic to construct the layer using RealType and input_shape.
 *     return my_layer<RealType, Extents...>{input_shape, output_features};
 *   }
 * };
 * ```
 *
 * @see layer for the requirements on the resulting layer type.
 * @see make_network for how build is used during network construction.
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

// Internal helper: Type trait to detect if a type is a sequential network
// which is a tuple of layers where all layers satisfy is_connectable.
template <class T>
struct is_sequential_network : public std::false_type {};

// Specialization matches tuples containing only layer types that are
// connectable.
template <layer... Ts>
  requires is_connectable<Ts...>::value
struct is_sequential_network<std::tuple<Ts...>> : public std::true_type {};

/**
 * @brief Specifies that a type qualifies as a sequential neural network.
 *
 * The concept sequential_network<T> specifies that T is the type of a
 * sequential neural network composed of two or more layers where each layer's
 * output shape matches the subsequent layer's input shape.
 *
 * ### Semantic requirements
 * sequential_network<T> is modeled only if:
 * - T is a specialization of `std::tuple` where each element models the layer
 *   concept
 * - T contains two or more layers
 * - For each adjacent pair of layers ($L_i$, $L_{i+1}$) in `T`:
 *   - $L_i::output\_extents == L_{i+1}::input\_extents$ (The output shape of a
 *     layer must exactly match the input shape of the next layer).
 *
 * ### Example
 * ```cpp
 * struct layer1 {
 *   using real_type = float;
 *   static constexpr std::extents<std::size_t, 2> input_extents{};
 *   static constexpr std::extents<std::size_t, 3> output_extents{};
 *   // ...
 * };
 * struct layer2 {
 *   using real_type = float;
 *   static constexpr std::extents<std::size_t, 3> input_extents{};
 *   static constexpr std::extents<std::size_t, 4> output_extents{};
 *   // ...
 * };
 *
 * // Matches: layer1 output [3] == layer2 input [3]
 * using my_network = std::tuple<layer1, layer2>;
 * static_assert(nou::sequential_network<my_network>);
 * ```
 *
 * @tparam T The type to be checked.
 * @see layer for the requirements on individual layers.
 * @see make_network for how sequential network is constructed.
 */
export template <class T>
concept sequential_network = is_sequential_network<T>::value;

/**
 * @brief Represents the type of the first layer in the given sequential network
 * type.
 *
 * The type alias first_layer_t<Network> extracts the type of the first layer in
 * Network if Network models the sequential_network concept. Otherwise, the
 * program is ill-formed.
 *
 * ### Example
 * ```cpp
 * using network = std::tuple<layer1, layer2, layer3>;
 * static_assert(std::same_as<nou::first_layer_t<network>, layer1>);
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model the
 * sequential_network concept.
 * @see sequential_network for the requirements on the Network type.
 */
export template <class Network>
  requires sequential_network<std::remove_cvref_t<Network>>
using first_layer_t = std::tuple_element_t<0, std::remove_cvref_t<Network>>;

/**
 * @brief Represents the type of the output layer in the given sequential
 * network type.
 *
 * The type alias output_layer_t<Network> extracts the type of the last layer in
 * Network if Network models the sequential_network concept. Otherwise, the
 * program is ill-formed.
 *
 * ### Example
 * ```cpp
 * using network = std::tuple<layer1, layer2, layer3>;
 * static_assert(std::same_as<nou::output_layer_t<network>, layer3>);
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model the
 * sequential_network concept.
 * @see sequential_network for the requirements on the Network type.
 */
export template <class Network>
  requires sequential_network<std::remove_cvref_t<Network>>
using output_layer_t =
    std::tuple_element_t<std::tuple_size_v<std::remove_cvref_t<Network>> - 1,
                         std::remove_cvref_t<Network>>;

/**
 * @brief Constructs a sequential neural network.
 *
 * The function template make_network constructs a sequential neural network by
 * composing the provided layers and/or layer builders. It automatically
 * resolves builders into concrete layers using the output properties of the
 * preceding layer.
 *
 * ### Example
 * ```cpp
 * struct layer1 { ... }; // models layer
 * struct layer2 { ... }; // models layer
 * struct layer3_builder {
 *   template <class RealType, class IndexType, std::size_t... Extents>
 *   auto build(std::extents<IndexType, Extents...> input_shape) const {
 *     return layer3<RealType, Extents...>{input_shape, /\* ... \*\/};
 *   }
 * };
 * // Construct a network: the third layer is built using properties from the
 * // second layer.
 * constexpr nou::sequential_network auto network =
 *     nou::make_network(layer1{}, layer2{}, layer3_builder{});
 * ```
 *
 * @tparam First The type of the first layer. This type must model the layer
 * concept.
 * @tparam Second The type of the second layer or layer builder. Let $L_1$ be
 * First, Second must satisfy one of:
 * - Second models the layer concept where Second::input_extents ==
 *   $L_1$::output_extents
 * - Second is a builder where `build<$L_1$::real_type>(second,
 *   $L_1$::output_extents)` results in a layer $L_2$ with `L_2::input_extents
 *   == $L_1$::output_extents`.
 * @tparam Rest The types of additional layers or builders. Each element $E_i$
 * must satisfy the same connectivity requirements with respect to its preceding
 * layer $L_{i-1}$.
 *
 * @param first The first layer of the network.
 * @param second The second layer or layer builder.
 * @param rest Additional layers or layer builders to be appended.
 * @return A sequential neural network that models the sequential_network
 * concept.
 * @note All arguments are taken by value and moved into the resulting network
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
