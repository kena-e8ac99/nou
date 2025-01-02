#pragma once

namespace nou {

template <class T>
concept layer = requires(T layer) {
  typename T::input_extents_type;
  typename T::output_extents_type;
};

}  // namespace nou

