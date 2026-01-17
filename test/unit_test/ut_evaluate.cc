#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <expected>
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

constexpr auto expected_metrics1 = 1.0;
constexpr auto metrics1 = []([[maybe_unused]] const auto& output,
                             [[maybe_unused]] const auto& target) -> double {
  return expected_metrics1;
};

constexpr nou::sequential_network auto network =
    nou::make_network(layer_input1_output2, layer_input2_output3);

constexpr nou::sequential_network auto network_errorable_with_error =
    nou::make_network(layer_input1_output2,
                      layer_input2_output3_errorable_with_error);

constexpr nou::sequential_network auto network_errorable_without_error =
    nou::make_network(layer_input1_output2,
                      layer_input2_output3_errorable_without_error);
}  // namespace

auto main() -> int {  // NOLINT(bugprone-exception-escape)
  // --- Test that evaluate forwards prediction output and target to loss
  //     function ---
  {
    constexpr auto loss_function_arguments = nou::evaluate(
        network, input1, target1, loss_function_return_arguments{});
    static_assert(std::get<0>(loss_function_arguments) ==
                  nou::predict(network, input1));
    static_assert(std::get<1>(loss_function_arguments) == target1);
  }

  // --- Test that evaluate returns loss function result for single input ---
  {
    static_assert(nou::evaluate(network, input1, target1, loss_function1) ==
                  expected_loss1);
  }

  // --- Test that evaluate rejects input with mismatched shape ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), decltype(invalid_input1),
                              decltype(target1), decltype(loss_function1)>);
  }

  // --- Test that evaluate rejects target with mismatched shape ---
  {
    static_assert(compile_fail_evaluate<decltype(network), decltype(input1),
                                        decltype(invalid_target1),
                                        decltype(loss_function1)>);
  }

  // --- Test that evaluate rejects loss function with invalid signature ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), std::array<double, 1>,
                              std::array<double, 3>,
                              // loss function with incorrect signature
                              decltype([](auto&&) -> void {})>);
  }

  // --- Test that evaluate propagates error from prediction ---
  {
    static_assert(nou::evaluate(network_errorable_with_error, input1, target1,
                                loss_function1)
                      .error() == nou::computation_error::invalid_input);
  }

  // --- Test that evaluate returns loss when prediction is errorable but
  //     succeeds ---
  {
    static_assert(nou::evaluate(network_errorable_without_error, input1,
                                target1, loss_function1)
                      .value() == expected_loss1);
  }

  // --- Test that evaluate computes loss over batch input ---
  {
    static_assert(nou::evaluate(network, batch_input1, batch_target1,
                                loss_function1) == expected_loss1);
  }

  // --- Test that evaluate rejects batch input with mismatched shape ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), decltype(invalid_input1),
                              decltype(batch_target1),
                              decltype(loss_function1)>);
  }

  // --- Test that evaluate rejects batch target with mismatched shape ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), decltype(batch_input1),
                              decltype(invalid_target1),
                              decltype(loss_function1)>);
  }

  // --- Test that evaluate rejects loss function with invalid signature for
  //     batch ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), decltype(batch_input1),
                              decltype(batch_target1),
                              // loss function with incorrect signature
                              decltype([](auto&&) -> void {})>);
  }

  // --- Test that evaluate propagates error from batch prediction ---
  {
    static_assert(nou::evaluate(network_errorable_with_error, batch_input1,
                                batch_target1, loss_function1)
                      .error() == nou::computation_error::invalid_input);
  }

  // --- Test that evaluate returns loss when batch prediction is errorable but
  //     succeeds ---
  {
    static_assert(nou::evaluate(network_errorable_without_error, batch_input1,
                                batch_target1, loss_function1)
                      .value() == expected_loss1);
  }

  // --- Test that evaluate falls back to element-wise invocation when loss
  //     function accepts only single inputs at compile time ---
  {
    static_assert(nou::evaluate(network, batch_input1, batch_target1,
                                loss_function_not_compatible_with_batch_3) ==
                  expected_loss1);
  }

  // --- Test that evaluate falls back to element-wise invocation when immutable
  //     loss function (called parallel unsequenced) accepts only single inputs
  //     at runtime ---
  {
    // NOLINTNEXTLINE
    assert(nou::evaluate(network, batch_input1, batch_target1,
                         loss_function_not_compatible_with_batch_3) ==
           expected_loss1);
  }

  // --- Test that evaluate falls back to element-wise invocation when mutable
  //     loss function accepts only single inputs at compile time ---
  {
    static_assert(
        nou::evaluate(network, batch_input1, batch_target1,
                      loss_function_not_compatible_with_batch_mutable_3) ==
        expected_loss1);
  }

  // --- Test that evaluate falls back to element-wise invocation when mutable
  //     loss function (called sequential) accepts only single inputs at
  //     runtime ---
  {
    auto loss_function = loss_function_not_compatible_with_batch_mutable_3;
    // NOLINTNEXTLINE
    assert(nou::evaluate(network, batch_input1, batch_target1, loss_function) ==
           expected_loss1);
  }

  // --- Test that evaluate handles empty batch input and target ---
  {
    constexpr std::array<std::array<double, 1>, 0> empty_batch_input{};
    constexpr std::array<std::array<double, 3>, 0> empty_batch_target{};
    static_assert(nou::evaluate(network, empty_batch_input, empty_batch_target,
                                loss_function1) == expected_loss1);
  }

  // --- Test that evaluate returns tuple of loss and metrics when both are
  //     provided ---
  {
    constexpr auto result =
        nou::evaluate(network, input1, target1, loss_function1, metrics1);
    static_assert(std::get<0>(result) == expected_loss1);
    static_assert(std::get<1>(result) == expected_metrics1);
  }

  // --- Test that evaluate returns tuple of loss and multiple metrics ---
  {
    constexpr auto result = nou::evaluate(network, input1, target1,
                                          loss_function1, metrics1, metrics1);
    static_assert(std::get<0>(result) == expected_loss1);
    static_assert(std::get<1>(result) == expected_metrics1);
    static_assert(std::get<2>(result) == expected_metrics1);
  }

  // --- Test that evaluate returns metrics directly when no loss function is
  //     provided ---
  {
    constexpr auto result = nou::evaluate(network, input1, target1, metrics1);
    static_assert(result == expected_metrics1);
  }

  // --- Test that evaluate returns tuple of multiple metrics when no loss
  //     function is provided ---
  {
    constexpr auto result =
        nou::evaluate(network, input1, target1, metrics1, metrics1);
    static_assert(std::get<0>(result) == expected_metrics1);
    static_assert(std::get<1>(result) == expected_metrics1);
  }

  // --- Test that evaluate returns tuple of loss and metrics for batch
  //     input ---
  {
    constexpr auto result = nou::evaluate(network, batch_input1, batch_target1,
                                          loss_function1, metrics1);
    static_assert(std::get<0>(result) == expected_loss1);
    static_assert(std::get<1>(result) == expected_metrics1);
  }

  // --- Test that evaluate returns tuple of loss and multiple metrics for batch
  //     input ---
  {
    constexpr auto result = nou::evaluate(network, batch_input1, batch_target1,
                                          loss_function1, metrics1, metrics1);
    static_assert(std::get<0>(result) == expected_loss1);
    static_assert(std::get<1>(result) == expected_metrics1);
    static_assert(std::get<2>(result) == expected_metrics1);
  }

  // --- Test that evaluate returns metrics directly for batch input when no
  //     loss function is provided ---
  {
    constexpr auto result =
        nou::evaluate(network, batch_input1, batch_target1, metrics1);
    static_assert(result == expected_metrics1);
  }

  // --- Test that evaluate returns tuple of multiple metrics for batch input
  //     when no loss function is provided ---
  {
    constexpr auto result =
        nou::evaluate(network, batch_input1, batch_target1, metrics1, metrics1);
    static_assert(std::get<0>(result) == expected_metrics1);
    static_assert(std::get<1>(result) == expected_metrics1);
  }

  // --- Test that evaluate rejects metric function with invalid signature ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), decltype(input1),
                              decltype(target1), decltype(loss_function1),
                              // metrics function with incorrect signature
                              decltype([](auto&&) -> void {})>);
  }

  // --- Test that evaluate rejects metric function with invalid signature for
  //     batch input ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), decltype(batch_input1),
                              decltype(batch_target1), decltype(loss_function1),
                              // metrics function with incorrect signature
                              decltype([](auto&&) -> void {})>);
  }

  // --- Test that evaluate rejects if no loss function and no metrics are
  //     provided ---
  {
    static_assert(compile_fail_evaluate<decltype(network), decltype(input1),
                                        decltype(target1)>);
  }

  // --- Test that evaluate rejects if no loss function and no metrics are
  //     provided in batch ---
  {
    static_assert(
        compile_fail_evaluate<decltype(network), decltype(batch_input1),
                              decltype(batch_target1)>);
  }

  return 0;
}