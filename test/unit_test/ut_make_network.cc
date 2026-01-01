#include <boost/ut.hpp>
#include <tuple>

import nou;

namespace {

template <class... Ts>
concept make_network_compile_success =
    requires(Ts... args) { nou::make_network(args...); };

struct layer {
  int value{};  // NOLINT(misc-non-private-member-variables-in-classes)
};

constexpr auto operator==(layer lhs, layer rhs) -> bool {
  return lhs.value == rhs.value;
}

constexpr auto is_same_instance(const layer& lhs, const layer& rhs) -> bool {
  return &lhs == &rhs;
}

constexpr nou::layer auto layer1 = layer{.value = 1};
constexpr nou::layer auto layer2 = layer{.value = 2};

}  // namespace

auto main() -> int {
  using namespace boost::ut;  // NOLINT(google-build-using-namespace)

  "construct model from two or more layers"_test = []() -> void {
    constexpr nou::network auto network = nou::make_network(layer1, layer2);

    static_assert(network == std::tuple{layer1, layer2});
    static_assert(not is_same_instance(std::get<0>(network), layer1));
    static_assert(not is_same_instance(std::get<1>(network), layer2));
  };

  "model construction fails if no arguments"_test = []() -> void {
    static_assert(not make_network_compile_success<>);
  };

  return 0;
}
