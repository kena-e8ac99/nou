#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <tuple>

namespace nou {

template <class T, std::unsigned_integral auto... Ns>
  requires(sizeof...(Ns) > 0)
class extents final {
 public:
  // Type aliases
  using element_type = T;
  using rank_type = std::size_t;
  using size_type = std::tuple_element_t<0, std::tuple<decltype(Ns)...>>;

  // Static Member functions
  static constexpr auto rank() noexcept -> rank_type { return sizeof...(Ns); }

  // Member functions
  static constexpr auto static_extent(rank_type rank) noexcept -> size_type {
    static constexpr std::array extents_{Ns...};
    return extents_[rank];
  }
};

}  // namespace nou

