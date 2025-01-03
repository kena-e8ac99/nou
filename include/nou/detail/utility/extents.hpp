#pragma once

#include <array>
#include <concepts>
#include <cstddef>
#include <optional>
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

  /**
 * @brief Returns the number of dimensions (rank) of the extents.
 *
 * @details This static member function computes the rank of the multi-dimensional array
 * by returning the number of template arguments provided in the parameter pack `Ns`.
 *
 * @return rank_type The number of dimensions in the extents.
 * @retval Number of template arguments representing array dimensions
 *
 * @note This function is marked `constexpr` and `noexcept`, allowing compile-time evaluation
 * and guaranteeing no exceptions will be thrown.
 *
 * @see sizeof...() operator for parameter pack size computation
 */
  static constexpr auto rank() noexcept -> rank_type { return sizeof...(Ns); }

  /**
   * @brief Retrieves the static extent for a given rank.
   *
   * Returns the extent (size) of a specific dimension in the multi-dimensional array.
   * 
   * @param rank The zero-based index of the dimension to retrieve.
   * @return std::optional<size_type> 
   *         - Contains the extent value if the rank is valid
   *         - Returns std::nullopt if the rank is out of bounds
   * 
   * @note Complexity: O(1) constant time access
   * @note Marked as constexpr, allowing compile-time evaluation
   * 
   * @throws Nothing (noexcept guarantee)
   * 
   * @example
   * extents<int, 3, 4, 5> ext;
   * auto dim1_size = ext.static_extent(1); // Returns 4
   * auto invalid_dim = ext.static_extent(3); // Returns std::nullopt
   */
  static constexpr auto static_extent(rank_type rank) noexcept
      -> std::optional<size_type> {
    if (rank >= extents<T, Ns...>::rank()) {
      return std::nullopt;
    }
    static constexpr std::array extents_{Ns...};
    return extents_[rank];
  }
};

}  // namespace nou

