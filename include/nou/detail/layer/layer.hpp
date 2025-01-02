#pragma once

#include <concepts>

namespace nou {

template <class T>
concept layer = requires(T layer) {
  typename T::input_extents_type;
  typename T::output_extents_type;

  { layer.input_extents() } -> std::same_as<typename T::input_extents_type>;
  { layer.output_extents() } -> std::same_as<typename T::output_extents_type>;
};

}  // namespace nou

