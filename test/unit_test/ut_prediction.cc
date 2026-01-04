#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <ranges>

import nou;

namespace {
template <class... Ts>
concept compile_fail_predict =
    not requires { nou::predict(std::declval<Ts>()...); };

template <auto InputSize, auto OutputSize, class RealType>
struct layer_propagate_member {
  using input_shape_type = nou::shape<InputSize>;
  using output_shape_type = nou::shape<OutputSize>;
  using real_type = RealType;
  std::array<real_type, InputSize>
      passed_input{};  // NOLINT(misc-non-private-member-variables-in-classes)
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate(Input&& input)
      -> std::array<real_type, OutputSize> {
    std::ranges::copy(std::forward<Input>(input), passed_input.begin());
    return output;
  }

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> std::array<real_type, OutputSize> {
    return output;
  }
};

constexpr std::array<double, 1> input1{1.0};
constexpr nou::layer auto layer_input1_output2_propagate_member =
    layer_propagate_member<1, 2, double>{.output = {2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_propagate_member =
    layer_propagate_member<2, 3, double>{.output = {4.0, 5.0, 6.0}};

template <auto InputSize, auto OutputSize, class RealType>
struct layer_propagate_free_func {
  using input_shape_type = nou::shape<InputSize>;
  using output_shape_type = nou::shape<OutputSize>;
  using real_type = RealType;
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)
};
template <auto InputSize, auto OutputSize, class RealType,
          std::ranges::input_range Input>
[[nodiscard]] constexpr auto propagate(
    const layer_propagate_free_func<InputSize, OutputSize, RealType>& layer,
    [[maybe_unused]] Input&& input) -> std::array<RealType, OutputSize> {
  return layer.output;
}
constexpr nou::layer auto layer_input1_output2_propagate_free_func =
    layer_propagate_free_func<1, 2, double>{.output = {2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_propagate_free_func =
    layer_propagate_free_func<2, 3, double>{.output = {4.0, 5.0, 6.0}};

template <auto InputSize, auto OutputSize, class RealType>
struct layer_propagate_hidden_friend {
  using input_shape_type = nou::shape<InputSize>;
  using output_shape_type = nou::shape<OutputSize>;
  using real_type = RealType;
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  friend constexpr auto propagate(const layer_propagate_hidden_friend& layer,
                                  [[maybe_unused]] Input&& input)
      -> std::array<RealType, OutputSize> {
    return layer.output;
  }
};
constexpr nou::layer auto layer_input1_output2_propagate_hidden_friend =
    layer_propagate_hidden_friend<1, 2, double>{.output = {2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_propagate_hidden_friend =
    layer_propagate_hidden_friend<2, 3, double>{.output = {4.0, 5.0, 6.0}};

template <auto InputSize, auto OutputSize, class RealType>
struct not_propagatable_layer {
  using input_shape_type = nou::shape<InputSize>;
  using output_shape_type = nou::shape<OutputSize>;
  using real_type = RealType;
};
constexpr auto not_propagatable_layer_input1_output2 =
    not_propagatable_layer<1, 2, double>{};
}  // namespace

auto main() -> int {
  using namespace boost::ut;  // NOLINT(google-build-using-namespace)

  "prediction propagates its input to the first layer"_test = []() -> void {
    constexpr auto passed_input_to_first_layer = []() -> auto {
      nou::network auto network =
          nou::make_network(layer_input1_output2_propagate_member,
                            layer_input2_output3_propagate_member);
      nou::predict(network, input1);
      return std::get<0>(network).passed_input;
    }();
    static_assert(passed_input_to_first_layer == input1);
  };

  "prediction propagates layer outputs to subsequent layers"_test =
      []() -> void {
    constexpr auto passed_input_to_second_layer = []() -> auto {
      nou::network auto network =
          nou::make_network(layer_input1_output2_propagate_member,
                            layer_input2_output3_propagate_member);
      nou::predict(network, input1);
      return std::get<1>(network).passed_input;
    }();
    static_assert(passed_input_to_second_layer ==
                  layer_input1_output2_propagate_member.output);
  };

  "prediction returns the output of the last layer"_test = []() -> void {
    constexpr nou::network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_member);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_member.output);
  };

  "predict network via propagate member function"_test = []() -> void {
    constexpr nou::network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_member);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_member.output);
  };

  "predict network via propagate free function"_test = []() -> void {
    constexpr nou::network auto network =
        nou::make_network(layer_input1_output2_propagate_free_func,
                          layer_input2_output3_propagate_free_func);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_free_func.output);
  };

  "predict network via propagate hidden friend function"_test = []() -> void {
    constexpr nou::network auto network =
        nou::make_network(layer_input1_output2_propagate_hidden_friend,
                          layer_input2_output3_propagate_hidden_friend);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_hidden_friend.output);
  };

  "predict const lvalue network"_test = []() -> void {
    constexpr nou::network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_member);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_member.output);
  };

  "predict rvalue network"_test = []() -> void {
    constexpr auto output =
        nou::predict(nou::make_network(layer_input1_output2_propagate_member,
                                       layer_input2_output3_propagate_member),
                     input1);
    static_assert(output == layer_input2_output3_propagate_member.output);
  };

  "prediction fails if layer cannot propagate"_test = []() -> void {
    static_assert(
        compile_fail_predict<decltype(nou::make_network(
                                 not_propagatable_layer_input1_output2,
                                 layer_input2_output3_propagate_member)),
                             decltype(input1)>);
  };

  return 0;
}