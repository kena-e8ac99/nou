#include <array>
#include <cassert>
#include <expected>
#include <mdspan>
#include <ranges>
#include <tuple>
#include <vector>

import nou;

namespace {
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

// batch_input1 = {{1.0}, {2.0}}
constexpr std::array<std::array<double, 1>, 2> batch_input1{
    std::array<double, 1>{1.0}, std::array<double, 1>{2.0}};

// batch_input2 = {{3.0}, {4.0}}
constexpr std::array<std::array<double, 1>, 2> batch_input2{
    std::array<double, 1>{3.0}, std::array<double, 1>{4.0}};

// batch_target1 = {{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}}
constexpr std::array<std::array<double, 3>, 2> batch_target1{
    std::array<double, 3>{2.0, 3.0, 4.0}, std::array<double, 3>{3.0, 4.0, 5.0}};

// batch_target2 = {{4.0, 5.0, 6.0}, {5.0, 6.0, 7.0}}
constexpr std::array<std::array<double, 3>, 2> batch_target2{
    std::array<double, 3>{4.0, 5.0, 6.0}, std::array<double, 3>{5.0, 6.0, 7.0}};

// batch_dataset1 = { (batch_input1, batch_target1) } aka
// { { {{1.0}, {2.0}}, {{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}} }
constexpr std::array<std::tuple<std::array<std::array<double, 1>, 2>,
                                std::array<std::array<double, 3>, 2>>,
                     1>
    batch_dataset1{std::tuple{batch_input1, batch_target1}};

// batch_dataset2_1 = { (batch_input1, batch_target1),
//                      (batch_input2, batch_target2) }
//  aka { { {{1.0}, {2.0}}, {{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}} },
//        { {{3.0}, {4.0}}, {{4.0, 5.0, 6.0}, {5.0, 6.0, 7.0}} } }
constexpr std::array<std::tuple<std::array<std::array<double, 1>, 2>,
                                std::array<std::array<double, 3>, 2>>,
                     2>
    batch_dataset2_1{std::tuple{batch_input1, batch_target1},
                     std::tuple{batch_input2, batch_target2}};

// batch_dataset2_2 = { (batch_input2, batch_target2),
//                      (batch_input1, batch_target1) }
//  aka { { {{3.0}, {4.0}}, {{4.0, 5.0, 6.0}, {5.0, 6.0, 7.0}} },
//        { {{1.0}, {2.0}}, {{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}} } }
constexpr std::array<std::tuple<std::array<std::array<double, 1>, 2>,
                                std::array<std::array<double, 3>, 2>>,
                     2>
    batch_dataset2_2{std::tuple{batch_input2, batch_target2},
                     std::tuple{batch_input1, batch_target1}};

// epoch_dataset1 = { batch_dataset1 } aka { { (batch_input1, batch_target1) } }
// aka { { { {{1.0}, {2.0}}, {{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}} } } }
constexpr std::array<
    std::array<std::tuple<std::array<std::array<double, 1>, 2>,
                          std::array<std::array<double, 3>, 2>>,
               1>,
    1>
    epoch_dataset1{batch_dataset1};

// epoch_dataset2 = { batch_dataset2_1, batch_dataset2_2 } aka
// { { (batch_input1, batch_target1), (batch_input2, batch_target2) },
//   { (batch_input2, batch_target2), (batch_input1, batch_target1) } }
// aka { { { {{1.0}, {2.0}}, {{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}} },
//         { {{3.0}, {4.0}}, {{4.0, 5.0, 6.0}, {5.0, 6.0, 7.0}} } },
//       { { {{3.0}, {4.0}}, {{4.0, 5.0, 6.0}, {5.0, 6.0, 7.0}} },
//         { {{1.0}, {2.0}}, {{2.0, 3.0, 4.0}, {3.0, 4.0, 5.0}} } } }
constexpr std::array<
    std::array<std::tuple<std::array<std::array<double, 1>, 2>,
                          std::array<std::array<double, 3>, 2>>,
               2>,
    2>
    epoch_dataset2{batch_dataset2_1, batch_dataset2_2};

struct optimizer {};
constexpr optimizer optimizers{};

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

constexpr nou::layer auto layer_input1_output2 =
    layer<1, 2, double, std::array<double, 2>>{{2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3 =
    layer<2, 3, double, std::array<double, 3>>{{4.0, 5.0, 6.0}};

constexpr nou::sequential_network auto network =
    nou::make_network(layer_input1_output2, layer_input2_output3);
}  // namespace

auto main() -> int {
  // --- Test that fit forwards prediction output and target to loss
  //     function for batch dataset ---
  {
    static_assert(
        (batch_dataset1 |
         nou::fit(network, optimizers, loss_function_return_arguments{}))
            .front() ==
        std::tuple{nou::predict(network, batch_input1), batch_target1});
  }

  // --- Test that fit forwards prediction output and target to loss
  //     function for batch dataset with multiple batches ---
  {
    constexpr auto fit_view =
        batch_dataset2_1 |
        nou::fit(network, optimizers, loss_function_return_arguments{});
    static_assert(
        fit_view.front() ==
        std::tuple{nou::predict(network, batch_input1), batch_target1});
    static_assert(
        fit_view.back() ==
        std::tuple{nou::predict(network, batch_input2), batch_target2});
  }

  // --- Test that fit returns loss function result for batch dataset ---
  {
    static_assert(
        (batch_dataset1 | nou::fit(network, optimizers, loss_function1))
            .front() == expected_loss1);
  }

  // --- Test that fit returns loss function result for batch dataset with
  //     multiple batches ---
  {
    constexpr auto fit_view =
        batch_dataset2_1 | nou::fit(network, optimizers, loss_function1);
    static_assert(fit_view.front() == expected_loss1);
    static_assert(fit_view.back() == expected_loss1);
  }

  // --- Test that fit works with direct call for batch dataset ---
  {
    static_assert(
        nou::fit(network, batch_dataset1, optimizers, loss_function1).front() ==
        expected_loss1);
  }

  // --- Test that fit returns loss function result for epoch dataset ---
  {
    static_assert(
        (epoch_dataset1 | nou::fit(network, optimizers, loss_function1))
            .front() == expected_loss1);
  }

  // --- Test that fit returns loss function result for epoch dataset with
  //     multiple epochs ---
  {
    constexpr auto fit_view =
        epoch_dataset2 | nou::fit(network, optimizers, loss_function1);
    static_assert(fit_view.front() == expected_loss1);
    static_assert(fit_view.back() == expected_loss1);
  }

  // --- Test that fit works with direct call for epoch dataset ---
  {
    static_assert(
        nou::fit(network, epoch_dataset1, optimizers, loss_function1).front() ==
        expected_loss1);
  }

  return 0;
}