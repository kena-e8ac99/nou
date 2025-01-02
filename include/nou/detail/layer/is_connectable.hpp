#pragma once

#include <concepts>
#include <type_traits>

#include "nou/detail/layer/layer.hpp"

namespace nou {

template <class... Ts>
struct is_connectable : std::false_type {};

template <layer T>
struct is_connectable<T> : std::true_type {};

template <layer T, layer U>
  requires std::same_as<typename T::output_extents_type,
                        typename U::input_extents_type>
struct is_connectable<T, U> : std::true_type {};

template <layer T, layer U, layer... Ts>
  requires std::same_as<typename T::output_extents_type,
                        typename U::input_extents_type>
struct is_connectable<T, U, Ts...> : is_connectable<U, Ts...> {};

template <class... Ts>
constexpr auto is_connectable_v = is_connectable<Ts...>::value;

}  // namespace nou

