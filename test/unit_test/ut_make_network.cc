#include <concepts>
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

template <auto InputSize, auto OutputSize>
constexpr auto operator==(layer<InputSize, OutputSize> lhs,
                          layer<InputSize, OutputSize> rhs) -> bool {
  return lhs.value == rhs.value;
}

template <auto InputSize, auto OutputSize>
constexpr auto is_same_reference(const layer<InputSize, OutputSize>& lhs,
                                 const layer<InputSize, OutputSize>& rhs)
    -> bool {
  return &lhs == &rhs;
}

using layer_input1_output2_t = layer<1, 2>;
using layer_input2_output3_t = layer<2, 3>;
using layer_input3_output4_t = layer<3, 4>;
constexpr nou::layer auto layer_input1_output2 =
    layer_input1_output2_t{.value = 1};
constexpr nou::layer auto layer_input2_output3 =
    layer_input2_output3_t{.value = 2};
constexpr nou::layer auto layer_input3_output4 =
    layer_input3_output4_t{.value = 3};

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

constexpr auto layer_builder_member_output3 =
    layer_builder_member<std::remove_cvref_t<
        decltype(layer_input2_output3)>::output_shape_type::value>{
        .value = layer_input2_output3.value};

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
constexpr auto layer_builder_free_func_output3 =
    layer_builder_free_func<std::remove_cvref_t<
        decltype(layer_input2_output3)>::output_shape_type::value>{
        .value = layer_input2_output3.value};

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
constexpr auto layer_builder_hidden_friend_output3 =
    layer_builder_hidden_friend<std::remove_cvref_t<
        decltype(layer_input2_output3)>::output_shape_type::value>{
        .value = layer_input2_output3.value};

struct not_layer_builder {};

struct incorrect_layer_builder {
  template <class RealType, class Shape>
  [[nodiscard]] constexpr auto build() const -> not_layer {
    return {};
  }
};
}  // namespace

auto main() -> int {
  // --- construct model from two layers ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);

    static_assert(network ==
                  std::tuple{layer_input1_output2, layer_input2_output3});
    static_assert(
        not is_same_reference(std::get<0>(network), layer_input1_output2));
    static_assert(
        not is_same_reference(std::get<1>(network), layer_input2_output3));
  }

  // --- construct model from three or more layers ---
  {
    constexpr nou::sequential_network auto network = nou::make_network(
        layer_input1_output2, layer_input2_output3, layer_input3_output4);

    static_assert(network == std::tuple{layer_input1_output2,
                                        layer_input2_output3,
                                        layer_input3_output4});
    static_assert(
        not is_same_reference(std::get<0>(network), layer_input1_output2));
    static_assert(
        not is_same_reference(std::get<1>(network), layer_input2_output3));
    static_assert(
        not is_same_reference(std::get<2>(network), layer_input3_output4));
  }

  // --- model construction fails if first argument is not layer ---
  {
    static_assert(compile_fail_make_network<not_layer, layer_input1_output2_t>);
  }

  // --- model construction fails if layer's input shape does not match ---
  {
    static_assert(compile_fail_make_network<layer_input1_output2_t,
                                            layer_input3_output4_t>);
  }

  // --- construct model from [layer, layer builder(member function)] ---
  {
    static_assert(
        nou::make_network(layer_input1_output2, layer_builder_member_output3) ==
        std::tuple{layer_input1_output2, layer_input2_output3});
  }

  // --- construct model from [layer, layer builder(free function)] ---
  {
    static_assert(nou::make_network(layer_input1_output2,
                                    layer_builder_free_func_output3) ==
                  std::tuple{layer_input1_output2, layer_input2_output3});
  }

  // --- construct model from [layer, layer builder(hidden friend function)] ---
  {
    static_assert(nou::make_network(layer_input1_output2,
                                    layer_builder_hidden_friend_output3) ==
                  std::tuple{layer_input1_output2, layer_input2_output3});
  }

  // --- construct model from [layer, layer builder, layer] ---
  {
    static_assert(nou::make_network(layer_input1_output2,
                                    layer_builder_member_output3,
                                    layer_input3_output4) ==
                  std::tuple{layer_input1_output2, layer_input2_output3,
                             layer_input3_output4});
  }

  // --- model construction fails if second is not layer or layer builder ---
  {
    static_assert(
        compile_fail_make_network<layer_input1_output2_t, not_layer_builder>);
  }

  // --- model construction fails if layer builder returns not a layer ---
  {
    static_assert(compile_fail_make_network<layer_input1_output2_t,
                                            incorrect_layer_builder>);
  }

  return 0;
}
