#include <boost/ut.hpp>
#include <tuple>
#include <type_traits>
#include <utility>

import nou;

namespace {

template <class... Ts>
concept compile_fail_make_network =
    not requires { nou::make_network(std::declval<Ts>()...); };

template <auto InputSize, auto OutputSize>
struct layer {
  using input_shape_type = nou::shape<InputSize>;
  using output_shape_type = nou::shape<OutputSize>;
  using real_type = double;
  real_type value{};  // NOLINT(misc-non-private-member-variables-in-classes)
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
constexpr nou::layer auto layer3 = layer<3, 4>{.value = 3};

struct not_layer {};

template <std::integral auto OutputSize>
struct layer_builder_member {
  template <class RealType, class Shape>
  [[nodiscard]] constexpr auto build() const
      -> layer<Shape::value, OutputSize> {
    return {.value = static_cast<RealType>(value)};
  }
  double value{};
};

constexpr auto layer_builder_member2 = layer_builder_member<
    std::remove_cvref_t<decltype(layer2)>::output_shape_type::value>{
    .value = layer2.value};

template <std::integral auto OutputSize>
struct layer_builder_free_func {
  double value{};
};

template <class RealType, class Shape, std::integral auto OutputSize>
[[nodiscard]] constexpr auto build(
    const layer_builder_free_func<OutputSize>& builder)
    -> layer<Shape::value, OutputSize> {
  return {.value = static_cast<RealType>(builder.value)};
}
constexpr auto layer_builder_free_func2 = layer_builder_free_func<
    std::remove_cvref_t<decltype(layer2)>::output_shape_type::value>{
    .value = layer2.value};

template <std::integral auto OutputSize>
struct layer_builder_hidden_friend {
  double value{};

  template <class RealType, class Shape>
  [[nodiscard]] friend constexpr auto build(
      const layer_builder_hidden_friend<OutputSize>& builder)
      -> layer<Shape::value, OutputSize> {
    return {.value = static_cast<RealType>(builder.value)};
  }
};
constexpr auto layer_builder_hidden_friend2 = layer_builder_hidden_friend<
    std::remove_cvref_t<decltype(layer2)>::output_shape_type::value>{
    .value = layer2.value};

struct not_layer_builder {};

struct incorrect_layer_builder {
  template <class RealType, class Shape>
  [[nodiscard]] constexpr auto build() const -> not_layer {
    return {};
  }
};
}  // namespace

auto main() -> int {
  using namespace boost::ut;  // NOLINT(google-build-using-namespace)

  "construct model from two layers"_test = []() -> void {
    constexpr nou::network auto network = nou::make_network(layer1, layer2);

    static_assert(network == std::tuple{layer1, layer2});
    static_assert(not is_same_instance(std::get<0>(network), layer1));
    static_assert(not is_same_instance(std::get<1>(network), layer2));
  };

  "construct model from three or more layers"_test = []() -> void {
    constexpr nou::network auto network =
        nou::make_network(layer1, layer2, layer3);

    static_assert(network == std::tuple{layer1, layer2, layer3});
    static_assert(not is_same_instance(std::get<0>(network), layer1));
    static_assert(not is_same_instance(std::get<1>(network), layer2));
    static_assert(not is_same_instance(std::get<2>(network), layer3));
  };

  "model construction fails if no arguments"_test = []() -> void {
    static_assert(compile_fail_make_network<>);
  };

  "model construction fails if only one layer is passed"_test = []() -> void {
    static_assert(compile_fail_make_network<layer<1, 2>>);
  };

  "model construction fails if first argument is not layer"_test =
      []() -> void {
    static_assert(compile_fail_make_network<not_layer, layer<1, 2>>);
  };

  "model construction fails if layer's input shape does not match"_test =
      []() -> void {
    static_assert(compile_fail_make_network<layer<1, 2>, layer<3, 4>>);
  };

  "construct model from [layer, layer builder(member function)]"_test =
      []() -> void {
    static_assert(nou::make_network(layer1, layer_builder_member2) ==
                  std::tuple{layer1, layer2});
  };

  "construct model from [layer, layer builder(free function)]"_test =
      []() -> void {
    static_assert(nou::make_network(layer1, layer_builder_free_func2) ==
                  std::tuple{layer1, layer2});
  };

  "construct model from [layer, layer builder(hidden friend function)]"_test =
      []() -> void {
    static_assert(nou::make_network(layer1, layer_builder_hidden_friend2) ==
                  std::tuple{layer1, layer2});
  };

  "construct model from [layer, layer builder, layer]"_test = []() -> void {
    static_assert(nou::make_network(layer1, layer_builder_member2, layer3) ==
                  std::tuple{layer1, layer2, layer3});
  };

  "model construction fails if second is not layer or layer builder"_test =
      []() -> void {
    static_assert(compile_fail_make_network<layer<1, 2>, not_layer_builder>);
  };

  "model construction fails if layer builder returns not a layer"_test =
      []() -> void {
    static_assert(
        compile_fail_make_network<layer<1, 2>, incorrect_layer_builder>);
  };

  return 0;
}
