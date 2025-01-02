#pragma once

#include "nou/detail/layer/layer.hpp"

namespace nou {

template <class T, class Extents>
concept builder = requires(T builder, Extents extents) {
  { builder.build(extents) } -> layer;
};

}  // namespace nou

