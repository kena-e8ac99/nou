#include "nou/detail/network/make_network.hpp"

#include <concepts>

#include "boost/ut.hpp"
#include "nou/detail/layer/layer.hpp"
#include "nou/detail/network/network.hpp"
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
[[nodiscard]] constexpr auto operator==(const layer<Ns...>&,
                                        const layer<Ns...>&) -> bool {
  return true;
}

template <std::unsigned_integral auto Output>
struct builder {
  using output_extents_type = nou::extents<float, Output>;

  template <class T, std::unsigned_integral auto Input>
  [[nodiscard]] constexpr auto build(
      const nou::extents<T, Input>&) const noexcept -> layer<Input, Output> {
    return {};
  }
};

}  // namespace mock

auto main() -> int {
  using namespace boost::ut;

  "make network from one layer"_test = [] {
    using layer = mock::layer<1UZ, 2UZ>;
    static_assert(nou::layer<layer>);

    constexpr auto network = nou::make_network(layer{});

    static_assert(std::same_as<decltype(auto(network)), nou::network<layer>>);
    static_assert(network == nou::network{layer{}});
  };

  "make network from two layers"_test = [] {
    using layer1 = mock::layer<1UZ, 2UZ>;
    using layer2 = mock::layer<2UZ, 3UZ>;
    static_assert(nou::layer<layer1> && nou::layer<layer2>);

    constexpr auto network = nou::make_network(layer1{}, layer2{});

    static_assert(
        std::same_as<decltype(auto(network)), nou::network<layer1, layer2>>);
    static_assert(network == nou::network{layer1{}, layer2{}});
  };

  "make network from three layers"_test = [] {
    using layer1 = mock::layer<1UZ, 2UZ>;
    using layer2 = mock::layer<2UZ, 3UZ>;
    using layer3 = mock::layer<3UZ, 4UZ>;
    static_assert(nou::layer<layer1> && nou::layer<layer2> &&
                  nou::layer<layer3>);

    constexpr auto network = nou::make_network(layer1{}, layer2{}, layer3{});

    static_assert(std::same_as<decltype(auto(network)),
                               nou::network<layer1, layer2, layer3>>);
    static_assert(network == nou::network{layer1{}, layer2{}, layer3{}});
  };

  "make network from one layer and one builder"_test = [] {
    using layer = mock::layer<1UZ, 2UZ>;
    using builder = mock::builder<3UZ>;
    static_assert(nou::layer<layer> &&
                  nou::builder<builder, typename layer::output_extents_type>);

    constexpr auto network = nou::make_network(layer{}, builder{});

    static_assert(std::same_as<decltype(auto(network)),
                               nou::network<layer, mock::layer<2UZ, 3UZ>>>);
    static_assert(network == nou::network{layer{}, mock::layer<2UZ, 3UZ>{}});
  };

  "make network from [layer -> builder -> layer]"_test = [] {
    using layer1 = mock::layer<1UZ, 2UZ>;
    using builder = mock::builder<3UZ>;
    using layer2 = mock::layer<3UZ, 4UZ>;
    static_assert(nou::layer<layer1> && nou::layer<layer2> &&
                  nou::builder<builder, typename layer1::output_extents_type>);

    constexpr auto network = nou::make_network(layer1{}, builder{}, layer2{});

    static_assert(
        std::same_as<decltype(auto(network)),
                     nou::network<layer1, mock::layer<2UZ, 3UZ>, layer2>>);
    static_assert(network ==
                  nou::network{layer1{}, mock::layer<2UZ, 3UZ>{}, layer2{}});
  };

  "make network from [layer -> builder -> builder]"_test = [] {
    using layer1 = mock::layer<1UZ, 2UZ>;
    using builder1 = mock::builder<3UZ>;
    using builder2 = mock::builder<4UZ>;
    static_assert(
        nou::layer<layer1> &&
        nou::builder<builder1, typename layer1::output_extents_type> &&
        nou::builder<builder2, typename builder1::output_extents_type>);

    constexpr auto network =
        nou::make_network(layer1{}, builder1{}, builder2{});

    static_assert(std::same_as<decltype(auto(network)),
                               nou::network<layer1, mock::layer<2UZ, 3UZ>,
                                            mock::layer<3UZ, 4UZ>>>);
    static_assert(network == nou::network{layer1{}, mock::layer<2UZ, 3UZ>{},
                                          mock::layer<3UZ, 4UZ>{}});
  };
}

