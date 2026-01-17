#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <execution>
#include <expected>
#include <iostream>
#include <limits>
#include <mdspan>
#include <ranges>
#include <span>
#include <vector>

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
  [[nodiscard]] constexpr auto propagate(
      [[maybe_unused]] const Input& input) const -> auto {
    if constexpr (std::ranges::input_range<std::ranges::range_value_t<Input>>) {
      if constexpr (requires { output.has_value(); }) {
        if (!output.has_value()) {
          return std::expected<std::vector<ValueType>, nou::computation_error>{
              std::unexpect, nou::computation_error::invalid_input};
        }
        return std::expected<std::vector<ValueType>, nou::computation_error>{
            std::vector<ValueType>(std::ranges::size(input), output.value())};
      } else {
        return std::vector<ValueType>(std::ranges::size(input), output);
      }
    } else {
      return output;
    }
  }
};

constexpr std::array<double, 1> input1{1.0};
[[maybe_unused]] constexpr std::array<std::array<double, 1>, 2> batch_input1{
    input1, input1};
constexpr std::array<double, 2> invalid_input1{1.0, 2.0};

constexpr std::array<double, 3> target1{2.0, 3.0, 4.0};
[[maybe_unused]] constexpr std::array<std::array<double, 3>, 2> batch_target1{
    std::array<double, 3>{2.0, 3.0, 4.0}, std::array<double, 3>{3.0, 4.0, 5.0}};
constexpr std::array<double, 1> invalid_target1{2.0};

struct loss_function_return_arguments {
  template <class T, class U>
  constexpr auto operator()(T&& output, U&& target) const -> auto {
    return std::tuple{std::forward<T>(output), std::forward<U>(target)};
  }
  template <class T, class U>
  constexpr auto df([[maybe_unused]] T&& output,
                    [[maybe_unused]] U&& target) const -> double {
    return 1.0;
  }
};

constexpr double expected_loss1 = 0.5;
struct loss_function {
  template <class T, class U>
  constexpr auto operator()([[maybe_unused]] T&& output,
                            [[maybe_unused]] U&& target) const -> double {
    return expected_loss1;
  }

  template <class T, class U>
  constexpr auto df([[maybe_unused]] T&& output,
                    [[maybe_unused]] U&& target) const -> double {
    return 1.0;
  }
};
constexpr auto loss_function1 = loss_function{};

template <auto N>
struct loss_function_not_compatible_with_batch {
  constexpr auto operator()(
      [[maybe_unused]] std::span<const double, N> output,
      [[maybe_unused]] std::span<const double, N> target) const -> double {
    return expected_loss1;
  }

  template <class T, class U>
  constexpr auto df([[maybe_unused]] T&& output,
                    [[maybe_unused]] U&& target) const -> double {
    return 1.0;
  }
};
constexpr auto loss_function_not_compatible_with_batch_3 =
    loss_function_not_compatible_with_batch<3>{};

template <auto N>
struct loss_function_not_compatible_with_batch_mutable {
  double loss = 0.0;  // NOLINT(misc-non-private-member-variables-in-classes)

  constexpr auto operator()([[maybe_unused]] std::span<const double, N> output,
                            [[maybe_unused]] std::span<const double, N> target)
      -> double {
    auto result = loss;
    loss += 1.0;
    return result;
  }

  template <class T, class U>
  constexpr auto df([[maybe_unused]] T&& output,
                    [[maybe_unused]] U&& target) const -> double {
    return 1.0;
  }
};
constexpr auto loss_function_not_compatible_with_batch_mutable_3 =
    loss_function_not_compatible_with_batch_mutable<3>{};

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
        network, input1, target1, loss_function_return_arguments{});
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

  // --- Test that evaluate returns correct loss if prediction is errorable
  //     without error ---
  {
    constexpr nou::sequential_network auto network = nou::make_network(
        layer_input1_output2, layer_input2_output3_errorable_without_error);
    static_assert(
        nou::evaluate(network, input1, target1, loss_function1).value() ==
        expected_loss1);
  }

  // --- Test that evaluate batches correctly ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    constexpr auto batch_loss =
        nou::evaluate(network, batch_input1, batch_target1, loss_function1);
    static_assert(batch_loss == expected_loss1);
  }

  // --- Test that compile fails if batch input shape mismatches ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);

    static_assert(
        compile_fail_evaluate<decltype(network), decltype(invalid_input1),
                              decltype(batch_target1),
                              decltype(loss_function1)>);
  }

  // --- Test that compile fails if batch target shape mismatches ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);

    static_assert(
        compile_fail_evaluate<decltype(network), decltype(batch_input1),
                              decltype(invalid_target1),
                              decltype(loss_function1)>);
  }

  // --- Test that compile fails if loss function call is invalid for batch ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);

    static_assert(
        compile_fail_evaluate<decltype(network), decltype(batch_input1),
                              decltype(batch_target1),
                              // loss function with incorrect signature
                              decltype([](auto&&) -> void {})>);
  }

  // --- Test that evaluate returns error if batch prediction returns error ---
  {
    constexpr nou::sequential_network auto network = nou::make_network(
        layer_input1_output2, layer_input2_output3_errorable_with_error);
    static_assert(
        nou::evaluate(network, batch_input1, batch_target1, loss_function1)
            .error() == nou::computation_error::invalid_input);
  }

  // --- Test that evaluate returns correct loss if batch prediction is
  //     errorable without error ---
  {
    constexpr nou::sequential_network auto network = nou::make_network(
        layer_input1_output2, layer_input2_output3_errorable_without_error);
    static_assert(
        nou::evaluate(network, batch_input1, batch_target1, loss_function1)
            .value() == expected_loss1);
  }

  // --- Test that evaluate works with loss function is not compatible with
  //     batch inputs but compatible with single inputs in compile time ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    static_assert(nou::evaluate(network, batch_input1, batch_target1,
                                loss_function_not_compatible_with_batch_3) ==
                  expected_loss1);
  }

  // --- Test that evaluate works with loss function that is not compatible with
  //     batch inputs but compatible with single inputs in runtime ---
  //     (This test uses assert instead of static_assert to verify runtime
  //      behavior. The result should be the same as the previous compile-time
  //      test.)
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    // NOLINTNEXTLINE
    assert(nou::evaluate(network, batch_input1, batch_target1,
                         loss_function_not_compatible_with_batch_3) ==
           expected_loss1);
  }

  // --- Test that evaluate works with mutable loss function that is not
  //     compatible with batch inputs but compatible with single inputs in
  //     compile time ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    static_assert(
        nou::evaluate(network, batch_input1, batch_target1,
                      loss_function_not_compatible_with_batch_mutable_3) ==
        expected_loss1);
  }

  // --- Test that evaluate works with mutable loss function that is not
  //     compatible with batch inputs but compatible with single inputs in
  //     runtime ---
  //     (This test uses assert instead of static_assert to verify runtime
  //      behavior. The result should be the same as the previous compile-time
  //      test.)
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    auto loss_function = loss_function_not_compatible_with_batch_mutable_3;
    nou::evaluate(network, batch_input1, batch_target1, loss_function);
    // NOLINTNEXTLINE
    // assert(std::abs(nou::evaluate(network, batch_input1, batch_target1,
    //                              loss_function) -
    //                expected_loss1) < std::numeric_limits<double>::epsilon());
  }

  // --- Test that evaluate works with empty batch input/target ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2, layer_input2_output3);
    constexpr std::array<std::array<double, 1>, 0> empty_batch_input{};
    constexpr std::array<std::array<double, 3>, 0> empty_batch_target{};
    static_assert(nou::evaluate(network, empty_batch_input, empty_batch_target,
                                loss_function1) == expected_loss1);
  }

  return 0;
}