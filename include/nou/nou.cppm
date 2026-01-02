module;

#include <concepts>
#include <tuple>
#include <utility>

export module nou;

namespace nou {

export template <std::integral auto N>
  requires(N > 0)
struct shape {};

template <class... Ts>
struct is_layer : public std::false_type {};

template <class T, std::integral auto InputSize, std::integral auto OutputSize>
  requires(InputSize > 0 and OutputSize > 0)
struct is_layer<T, shape<InputSize>, shape<OutputSize>>
    : public std::true_type {};

template <class T>
  requires requires {
    typename T::input_shape_type;
    typename T::output_shape_type;
  }
struct is_layer<T> : public is_layer<T, typename T::input_shape_type,
                                     typename T::output_shape_type> {};

export template <class T>
concept layer = is_layer<T>::value;

template <class T>
constexpr bool is_network_v = false;

template <layer... Ts>
constexpr bool is_network_v<std::tuple<Ts...>> = true;

export template <class T>
concept network = is_network_v<T>;

template <class... Ts>
struct is_same_shape : public std::false_type {};

template <std::integral auto N, std::integral auto M>
  requires(static_cast<std::common_type_t<decltype(N), decltype(M)>>(N) ==
           static_cast<std::common_type_t<decltype(N), decltype(M)>>(M))
struct is_same_shape<shape<N>, shape<M>> : public std::true_type {};

template <class... Ts>
struct is_connectable : public std::false_type {};

template <layer Layer>
struct is_connectable<Layer> : public std::true_type {};

template <layer First, layer Second, layer... Rest>
  requires is_same_shape<typename First::output_shape_type,
                         typename Second::input_shape_type>::value
struct is_connectable<First, Second, Rest...>
    : public is_connectable<Second, Rest...> {};

export template <layer T, layer... Ts>
  requires(sizeof...(Ts) >= 1) and is_connectable<T, Ts...>::value
constexpr auto make_network(T first, Ts... layers) -> std::tuple<T, Ts...> {
  return std::tuple{std::move(first), std::move(layers)...};
}

}  // namespace nou
