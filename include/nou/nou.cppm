module;

#include <utility>
#include <tuple>

export module nou;

namespace nou {

export template <class T>
concept layer = true;

template <class T>
constexpr bool is_network_v = false;

template <layer... Ts>
constexpr bool is_network_v<std::tuple<Ts...>> = true;

export template <class T>
concept network = is_network_v<T>;

export template <class... Ts>
requires (sizeof...(Ts) != 0)
constexpr auto make_network(Ts... layers) -> std::tuple<Ts...> {
  return std::tuple{std::move(layers)...};
}

} // namespace nou
