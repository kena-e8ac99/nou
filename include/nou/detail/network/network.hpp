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
class network final {
 public:
  /**
 * @brief Default constructor for the network class.
 * 
 * Creates an empty network with no layers. This constructor allows creating a 
 * network object without initially specifying any layers.
 * 
 * @note The network can be populated with layers later if needed.
 */
  network() = default;

  template <class... Ts>
  /**
       * @brief Constructs a network with the given layers.
       *
       * @tparam Ts Variadic template parameter types of the layers
       * @param layers Rvalue references to layers to be added to the network
       *
       * @note Uses perfect forwarding to efficiently construct the internal tuple of layers
       * @note Marked noexcept to indicate no exceptions will be thrown during construction
       * @note Requires all layers to be connectable via is_connectable_v concept
       *
       * @details This constructor creates a network by forwarding the provided layers 
       * into an internal std::tuple, preserving their value categories and avoiding unnecessary copies.
       */
      [[nodiscard]] explicit constexpr network(Ts&&... layers) noexcept
      : layers_{std::make_tuple(std::forward<Ts>(layers)...)} {}

  // Operator overloads
  template <class... Ts>
    requires(std::equality_comparable<Layers> && ...)
  /**
   * @brief Compares two network instances for equality.
   *
   * Checks if the layers of two network instances are equal.
   *
   * @tparam Ts Parameter pack representing the layer types of the network
   * @param rhs The right-hand side network to compare against
   * @return bool True if all layers are equal, false otherwise
   * @note This operation is noexcept if the underlying layer comparison is noexcept
   * @note Uses std::tuple comparison to check layer equality
   */
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

