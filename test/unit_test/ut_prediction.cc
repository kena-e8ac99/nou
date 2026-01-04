#include <algorithm>
#include <array>
#include <boost/ut.hpp>
#include <ranges>

import nou;

namespace {
template <auto InputSize, auto OutputSize, class RealType>
struct layer {
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
constexpr nou::layer auto layer_input1_output2 =
    layer<1, 2, double>{.output = {2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3 =
    layer<2, 3, double>{.output = {4.0, 5.0, 6.0}};
[[maybe_unused]] constexpr nou::layer auto layer_input3_output4 =
    layer<3, 4, double>{.output = {7.0, 8.0, 9.0, 10.0}};
}  // namespace

auto main() -> int {
  using namespace boost::ut;  // NOLINT(google-build-using-namespace)

  "prediction propagates its input to the first layer"_test = []() -> void {
    constexpr auto passed_input_to_first_layer = []() -> auto {
      nou::network auto network =
          nou::make_network(layer_input1_output2, layer_input2_output3);
      nou::predict(network, input1);
      return std::get<0>(network).passed_input;
    }();
    static_assert(passed_input_to_first_layer == input1);
  };

  "prediction propagates layer outputs to subsequent layers"_test =
      []() -> void {
    constexpr auto passed_input_to_second_layer = []() -> auto {
      nou::network auto network =
          nou::make_network(layer_input1_output2, layer_input2_output3);
      nou::predict(network, input1);
      return std::get<1>(network).passed_input;
    }();
    static_assert(passed_input_to_second_layer == layer_input1_output2.output);
  };

  "prediction returns the output of the last layer"_test = []() -> void {
    constexpr nou::network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    static_assert(nou::predict(network, input1) == layer_input2_output3.output);
  };

  return 0;
}