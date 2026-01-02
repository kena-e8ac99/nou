#include <boost/ut.hpp>
#include <tuple>
#include <utility>

import nou;

namespace {

template <class... Ts>
concept make_network_compile_success =
    requires { nou::make_network(std::declval<Ts>()...); };

template <auto InputSize, auto OutputSize>
struct layer {
  using input_shape_type = nou::shape<InputSize>;
  using output_shape_type = nou::shape<OutputSize>;
  int value{};  // NOLINT(misc-non-private-member-variables-in-classes)
};

template <auto N, auto M>
constexpr auto operator==(layer<N, M> lhs, layer<N, M> rhs) -> bool {
  return lhs.value == rhs.value;
}

template <auto N, auto M>
constexpr auto is_same_instance(const layer<N, M>& lhs, const layer<N, M>& rhs)
    -> bool {
  return &lhs == &rhs;
}

constexpr nou::layer auto layer1 = layer<1, 2>{.value = 1};
constexpr nou::layer auto layer2 = layer<2, 3>{.value = 2};

struct not_layer {};

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

  "model construction fails if only one layer is passed"_test = []() -> void {
    static_assert(not make_network_compile_success<layer<1, 2>>);
  };

  "model construction fails if first argument is not layer"_test =
      []() -> void {
    static_assert(not make_network_compile_success<not_layer, layer<1, 2>>);
  };

  "model construction fails if layer's input/output shape does not match"_test =
      []() -> void {
    static_assert(not make_network_compile_success<layer<1, 2>, layer<3, 4>>);
  };

  return 0;
}
