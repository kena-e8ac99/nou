#pragma once

#include <tuple>
#include <utility>

#include "nou/detail/layer/builder.hpp"
#include "nou/detail/layer/layer.hpp"
#include "nou/detail/network/network.hpp"

namespace nou {

template <layer Layer>
[[nodiscard]] constexpr auto make_network(Layer&& layer) noexcept {
  return network{std::forward<Layer>(layer)};
}

template <layer Layer, class... Ts>
[[nodiscard]] constexpr auto make_network(Layer&& first, Ts&&... trailing) {
  return []<class Self, class Layers, class T, class... Us>(
             this Self&& self, Layers&& layers, T&& target, Us&&... trailing)
    requires layer<decltype(auto(target))> ||
             builder<decltype(auto(target)),
                     typename decltype(auto(
                         std::get<std::tuple_size_v<Layers> - 1>(
                             layers)))::output_extents_type>
  {
    auto concat = [](auto&& layers, auto&& target) {
      if constexpr (::nou::layer<decltype(auto(target))>) {
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

