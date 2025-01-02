#include "nou/detail/network/network.hpp"

#include <concepts>

#include "boost/ut.hpp"
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

template <std::unsigned_integral auto... Ns>
constexpr auto operator==(const layer<Ns...>&, const layer<Ns...>&) -> bool {
  return true;
}

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  "concepts"_test = [] {
    using layer1 = mock::layer<1UZ, 2UZ>;
    using layer2 = mock::layer<2UZ, 3UZ>;
    using network = nou::network<layer1, layer2>;

    static_assert(std::regular<network>);
    static_assert(std::constructible_from<network, layer1, layer2>);
    static_assert(std::constructible_from<network, layer1&, layer2&>);
    static_assert(
        std::constructible_from<network, const layer1&, const layer2&>);
    static_assert(std::constructible_from<network, layer1&&, layer2&&>);
  };

  [[maybe_unused]] suite constructor = [] {
    "network should be constructed with one layer"_test = [] {
      using layer_t = mock::layer<1UZ, 1UZ>;
      static_assert(nou::network<layer_t>{} == nou::network{layer_t{}});
    };

    "network should be constructed with connectable layers"_test = [] {
      using layer1_t = mock::layer<1UZ, 2UZ>;
      using layer2_t = mock::layer<2UZ, 3UZ>;
      static_assert(nou::is_connectable_v<layer1_t, layer2_t>);

      static_assert(nou::network<layer1_t, layer2_t>{} ==
                    nou::network{layer1_t{}, layer2_t{}});
    };
  };

  return 0;
}

