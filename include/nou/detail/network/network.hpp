#pragma once

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

#include "nou/detail/layer/is_connectable.hpp"
#include "nou/detail/layer/layer.hpp"

namespace nou {

template <layer... Layers>
  requires is_connectable_v<Layers...>
class network {
 public:
  // Constructors
  network() = default;

  template <class... Ts>
  explicit constexpr network(Ts&&... layers) noexcept
      : layers_{std::make_tuple(std::forward<Ts>(layers)...)} {}

  // Operator overloads
  template <class... Ts>
    requires(std::equality_comparable<Layers> && ...)
  [[nodiscard]] constexpr auto operator==(const network<Ts...>& rhs) const
      noexcept(noexcept(layers_ == rhs.layers_)) -> bool {
    return layers_ == rhs.layers_;
  }

 private:
  // Type aliases
  using layers_type = std::tuple<Layers...>;

  // Member variables
  layers_type layers_;
};

template <class... Ts>
network(Ts&&...) -> network<std::remove_cvref_t<Ts>...>;

}  // namespace nou

