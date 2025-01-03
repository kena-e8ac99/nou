#pragma once

#include <tuple>
#include <utility>

#include "nou/detail/layer/builder.hpp"
#include "nou/detail/layer/layer.hpp"
#include "nou/detail/network/network.hpp"

namespace nou {

template <layer Layer>
/**
 * @brief Creates a network with a single layer.
 *
 * Constructs a network object by perfectly forwarding the provided layer.
 * This overload is optimized for creating a network from one layer.
 *
 * @tparam Layer The type of the layer used to construct the network
 * @param layer The layer to be used in network construction
 * @return A network object containing the provided layer
 * @note The function is marked noexcept, guaranteeing no exception will be thrown
 * @note Uses perfect forwarding to preserve the value category of the input layer
 */
[[nodiscard]] constexpr auto make_network(Layer&& layer) noexcept {
  return network{std::forward<Layer>(layer)};
}

template <layer Layer, class... Ts>
/**
 * @brief Constructs a network from one or more layers or builders.
 *
 * This function allows flexible network creation by accepting a single layer or multiple layers/builders.
 * It supports both direct layer instantiation and builder-based layer construction.
 *
 * @tparam Layer The type of the first layer or builder
 * @tparam Ts Types of additional layers or builders
 * @param first The first layer or builder to include in the network
 * @param trailing Additional layers or builders to include in the network
 *
 * @return A network constructed from the provided layers
 *
 * @note The function uses perfect forwarding and type deduction to handle various layer types
 * @note Builders are automatically converted to layers using their output extents
 * @note The return value must be used and cannot be discarded
 *
 * @throws No exceptions are thrown during network construction
 *
 * @example
 * // Creating a network with a single layer
 * auto net1 = make_network(dense_layer{...});
 *
 * // Creating a network with multiple layers
 * auto net2 = make_network(input_layer{...}, dense_layer{...}, output_layer{...});
 *
 * // Creating a network with builders
 * auto net3 = make_network(input_builder{...}, dense_builder{...}, output_builder{...});
 */
[[nodiscard]] constexpr auto make_network(Layer&& first, Ts&&... trailing) {
  return []<class Self, class Layers, class T, class... Us>(
             this Self&& self, Layers&& layers, T&& target, Us&&... trailing)
    requires layer<decltype(auto(target))> ||
             builder<decltype(auto(target)),
                     decltype(std::get<std::tuple_size_v<Layers> - 1>(layers)
                                  .output_extents())>
  {
    auto concat = [](auto&& layers, auto&& target) {
      if constexpr (layer<decltype(auto(target))>) {
        return std::tuple_cat(std::forward<Layers>(layers),
                              std::tuple{std::forward<T>(target)});
      } else {
        auto layer = std::forward<T>(target).build(
            std::get<std::tuple_size_v<Layers> - 1>(layers).output_extents());
        return std::tuple_cat(std::forward<Layers>(layers),
                              std::tuple{std::move(layer)});
      }
    }(std::forward<Layers>(layers), std::forward<T>(target));

    if constexpr (sizeof...(Us) == 0) {
      return std::apply(
          []<class... Vs>(Vs&&... layers) noexcept {
            return network{std::forward<Vs>(layers)...};
          },
          std::move(concat));
    } else {
      return std::forward<Self>(self)(std::move(concat),
                                      std::forward<Us>(trailing)...);
    }
  }(std::make_tuple(std::forward<Layer>(first)), std::forward<Ts>(trailing)...);
}

}  // namespace nou

