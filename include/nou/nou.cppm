module;

#include <concepts>
#include <tuple>
#include <type_traits>
#include <utility>

export module nou;

namespace nou {

export template <std::integral auto N>
  requires(N > 0)
struct shape {
  using type = decltype(N);
  static constexpr type value = N;
};

export template <class T>
concept layer = requires {
  typename T::real_type;
  typename T::input_shape_type;
  typename T::output_shape_type;
  requires std::same_as<typename T::input_shape_type,
                        shape<T::input_shape_type::value>>;
  requires std::same_as<typename T::output_shape_type,
                        shape<T::output_shape_type::value>>;
};

template <class T>
constexpr bool is_network_v = false;

template <layer... Ts>
constexpr bool is_network_v<std::tuple<Ts...>> = true;

export template <class T>
concept network = is_network_v<T>;

struct build_impl {
  template <class RealType, class Shape, class T>
    requires requires {
      { std::declval<T>().template build<RealType, Shape>() } -> layer;
    }
  constexpr auto operator()(T&& builder) const noexcept(
      noexcept(std::forward<T>(builder).template build<RealType, Shape>()))
      -> layer auto {
    return std::forward<T>(builder).template build<RealType, Shape>();
  }

  template <class RealType, class Shape, class T>
    requires(
        !requires { std::declval<T>().template build<RealType, Shape>(); } &&
        requires {
          { build<RealType, Shape>(std::declval<T>()) } -> layer;
        })
  constexpr auto operator()(T&& builder) const
      noexcept(noexcept(build<RealType, Shape>(std::forward<T>(builder))))
          -> layer auto {
    return build<RealType, Shape>(std::forward<T>(builder));
  }
};
inline namespace cpo {
inline constexpr auto build = build_impl{};
}

template <class... Ts>
struct is_connectable : public std::false_type {};

template <layer Layer>
struct is_connectable<Layer> : public std::true_type {};

template <layer First, layer Second, class... Rest>
  requires(
      static_cast<std::common_type_t<typename First::output_shape_type::type,
                                     typename Second::input_shape_type::type>>(
          First::output_shape_type::value) ==
      static_cast<std::common_type_t<typename First::output_shape_type::type,
                                     typename Second::input_shape_type::type>>(
          Second::input_shape_type::value))
struct is_connectable<First, Second, Rest...>
    : public is_connectable<Second, Rest...> {};

template <layer First, class Second, class... Rest>
  requires requires {
    {
      build.template
      operator()<typename First::real_type, typename First::output_shape_type>(
          std::declval<Second>())
    } -> layer;
  }
struct is_connectable<First, Second, Rest...>
    : public is_connectable<
          First,
          decltype(build.template operator()<typename First::real_type,
                                             typename First::output_shape_type>(
              std::declval<Second>())),
          Rest...> {};

export template <layer First, class Second, class... Rest>
  requires(is_connectable<First, Second, Rest...>::value)
constexpr auto make_network(First first, Second second, Rest... rest) -> network
    auto {
  constexpr auto builder = []<std::size_t I, class T>(T& tuple) -> layer auto {
    if constexpr (layer<std::tuple_element_t<I, T>>) {
      return std::move(std::get<I>(tuple));
    } else {
      using prev_layer_type = std::tuple_element_t<I - 1, T>;
      using real_type = prev_layer_type::real_type;
      using input_shape_type = prev_layer_type::output_shape_type;

      return build.template operator()<real_type, input_shape_type>(
          std::move(std::get<I>(tuple)));
    }
  };
  return [&]<std::size_t... Is>(auto tuple, std::index_sequence<Is...>)
             -> network auto {
    return std::tuple{[&]() -> layer auto {
      return builder.template operator()<Is>(tuple);
    }()...};
  }(std::tuple{std::move(first), std::move(second), std::move(rest)...},
             std::make_index_sequence<sizeof...(Rest) + 2>{});
}

}  // namespace nou
