#include "nou/detail/layer/is_connectable.hpp"

#include <boost/ut.hpp>
#include <concepts>

#include "nou/detail/utility/extents.hpp"

namespace mock {

template <std::unsigned_integral auto Input, std::unsigned_integral auto Output>
struct layer {
  using input_extents_type = nou::extents<float, Input>;
  using output_extents_type = nou::extents<float, Output>;

  [[nodiscard]] constexpr auto input_extents() const noexcept
      -> input_extents_type {
    return {};
  }
  [[nodiscard]] constexpr auto output_extents() const noexcept
      -> output_extents_type {
    return {};
  }
};

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  "one layer should be connectable"_test = [] {
    static_assert(nou::is_connectable_v<mock::layer<1UZ, 1UZ>>);
  };

  "two layers with matching extents should be connectable"_test = [] {
    static_assert(
        nou::is_connectable_v<mock::layer<1UZ, 2UZ>, mock::layer<2UZ, 3UZ>>);
  };

  "two layers with non-matching extents should not be connectable"_test = [] {
    static_assert(not nou::is_connectable_v<mock::layer<1UZ, 2UZ>,
                                            mock::layer<3UZ, 4UZ>>);
  };

  "three layers with matching extents should be connectable"_test = [] {
    static_assert(
        nou::is_connectable_v<mock::layer<1UZ, 2UZ>, mock::layer<2UZ, 3UZ>,
                              mock::layer<3UZ, 4UZ>>);
  };

  "three layers with non-matching extents should not be connectable"_test = [] {
    static_assert(
        not nou::is_connectable_v<mock::layer<1UZ, 2UZ>, mock::layer<2UZ, 3UZ>,
                                  mock::layer<4UZ, 3UZ>>);
  };
}

