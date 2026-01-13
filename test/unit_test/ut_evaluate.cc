#include <algorithm>
#include <array>
#include <mdspan>

import nou;

namespace {
template <auto InputSize, auto OutputSize, class RealType>
struct layer {
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
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
constexpr std::array<double, 3> target1{2.0, 3.0, 4.0};
constexpr nou::layer auto layer_input1_output2 =
    layer<1, 2, double>{.output = {2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3 =
    layer<2, 3, double>{.output = {4.0, 5.0, 6.0}};
}  // namespace

auto main() -> int {
  // --- Test that evaluate forwards correct output and target to loss function
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    [[maybe_unused]] constexpr auto loss_function_arguments = nou::evaluate(
        network, input1, target1,
        []<class T, class U>(T&& output, U&& target) -> auto {
          return std::tuple{std::forward<T>(output), std::forward<U>(target)};
        });
    static_assert(std::get<0>(loss_function_arguments) ==
                  nou::predict(network, input1));
    static_assert(std::get<1>(loss_function_arguments) == target1);
  }

  // --- Test that evaluate returns correct loss function result ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    constexpr double expected_loss_value = 42.0;
    constexpr auto loss_value = nou::evaluate(
        network, input1, target1,
        []<class T, class U>(T&& /*output*/, U&& /*target*/) -> double {
          return expected_loss_value;
        });
    static_assert(loss_value == expected_loss_value);
  }
  return 0;
}