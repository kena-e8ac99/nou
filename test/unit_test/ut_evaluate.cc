#include <array>
#include <expected>
#include <mdspan>

import nou;

namespace {
template <class... Ts>
concept compile_fail_evaluate =
    not requires { nou::evaluate(std::declval<Ts>()...); };

template <auto InputSize, auto OutputSize, class RealType, class ValueType>
struct layer {
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
  ValueType output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> ValueType {
    return output;
  }
};

constexpr std::array<double, 1> input1{1.0};
constexpr std::array<double, 2> invalid_input1{1.0, 2.0};
constexpr std::array<double, 3> target1{2.0, 3.0, 4.0};
constexpr std::array<double, 1> invalid_target1{2.0};
constexpr double expected_loss1 = 0.5;
constexpr auto loss_function1 = []([[maybe_unused]] auto&& output,
                                   [[maybe_unused]] auto&& target) -> double {
  return expected_loss1;
};

constexpr nou::layer auto layer_input1_output2 =
    layer<1, 2, double, std::array<double, 2>>{{2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3 =
    layer<2, 3, double, std::array<double, 3>>{{4.0, 5.0, 6.0}};
constexpr nou::layer auto layer_input2_output3_errorable_with_error =
    layer<2, 3, double,
          std::expected<std::array<double, 3>, nou::computation_error>>{
        std::expected<std::array<double, 3>, nou::computation_error>{
            std::unexpect, nou::computation_error::invalid_input}};
constexpr nou::layer auto layer_input2_output3_errorable_without_error =
    layer<2, 3, double,
          std::expected<std::array<double, 3>, nou::computation_error>>{
        std::array<double, 3>{4.0, 5.0, 6.0}};

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
    constexpr auto loss_value =
        nou::evaluate(network, input1, target1, loss_function1);
    static_assert(loss_value == expected_loss1);
  }

  // --- Test that compile fails if input shape mismatches ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);

    static_assert(
        compile_fail_evaluate<decltype(network), decltype(invalid_input1),
                              decltype(target1), decltype(loss_function1)>);
  }

  // --- Test that compile fails if target shape mismatches ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);

    static_assert(compile_fail_evaluate<decltype(network), decltype(input1),
                                        decltype(invalid_target1),
                                        decltype(loss_function1)>);
  }

  // --- Test that compile fails if loss function call is invalid ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);

    static_assert(
        compile_fail_evaluate<decltype(network), std::array<double, 1>,
                              std::array<double, 3>,
                              // loss function with incorrect signature
                              decltype([](auto&&) -> void {})>);
  }

  // --- Test that evaluate returns error if prediction returns error ---
  {
    constexpr nou::sequential_network auto network = nou::make_network(
        layer_input1_output2, layer_input2_output3_errorable_with_error);
    static_assert(
        nou::evaluate(network, input1, target1, loss_function1).error() ==
        nou::computation_error::invalid_input);
  }

  // --- Test that evaluate works if prediction is errorable but no error ---
  {
    constexpr nou::sequential_network auto network = nou::make_network(
        layer_input1_output2, layer_input2_output3_errorable_without_error);
    static_assert(
        nou::evaluate(network, input1, target1, loss_function1).value() ==
        expected_loss1);
  }

  return 0;
}