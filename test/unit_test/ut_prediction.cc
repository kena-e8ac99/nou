#include <algorithm>
#include <array>
#include <expected>
#include <mdspan>
#include <ranges>
#include <vector>

import nou;

namespace {
template <class... Ts>
concept compile_fail_predict =
    not requires { nou::predict(std::declval<Ts>()...); };

template <auto InputSize, auto OutputSize, class RealType>
struct layer_propagate_member {
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
constexpr nou::layer auto layer_input1_output2_propagate_member =
    layer_propagate_member<1, 2, double>{.output = {2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_propagate_member =
    layer_propagate_member<2, 3, double>{.output = {4.0, 5.0, 6.0}};
constexpr nou::layer auto layer_input3_output4_propagate_member =
    layer_propagate_member<3, 4, double>{.output = {7.0, 8.0, 9.0, 10.0}};

template <auto InputSize, auto OutputSize, class RealType>
struct layer_propagate_free_func {
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
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
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
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
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
};
constexpr auto not_propagatable_layer_input1_output2 =
    not_propagatable_layer<1, 2, double>{};

template <auto InputSize, auto OutputSize>
struct computation_error_layer {
  using real_type = double;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
  std::expected<std::array<real_type, OutputSize>, nou::computation_error>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> std::expected<std::array<real_type, OutputSize>,
                       nou::computation_error> {
    return output;
  }
};
constexpr nou::layer auto errorable_layer_input1_output2_with_error =
    computation_error_layer<1, 2>{
        .output = std::unexpected{nou::computation_error::invalid_input}};
constexpr nou::layer auto errorable_layer_input2_output3_with_error =
    computation_error_layer<2, 3>{
        .output = std::unexpected{nou::computation_error::invalid_input}};
constexpr nou::layer auto errorable_layer_input3_output4_with_error =
    computation_error_layer<3, 4>{
        .output = std::unexpected{nou::computation_error::invalid_input}};
constexpr nou::layer auto errorable_layer_input1_output2_without_error =
    computation_error_layer<1, 2>{.output = std::array<double, 2>{2.0, 3.0}};
constexpr nou::layer auto errorable_layer_input2_output3_without_error =
    computation_error_layer<2, 3>{.output =
                                      std::array<double, 3>{4.0, 5.0, 6.0}};
constexpr nou::layer auto errorable_layer_input3_output4_without_error =
    computation_error_layer<3, 4>{
        .output = std::array<double, 4>{7.0, 8.0, 9.0, 10.0}};

template <auto InputSize, auto OutputSize, class RealType>
struct layer_propagate_vector {
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> std::vector<real_type> {
    return std::ranges::to<std::vector<real_type>>(output);
  }
};
constexpr nou::layer auto layer_input1_output2_propagate_vector =
    layer_propagate_vector<1, 2, double>{{2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_propagate_vector =
    layer_propagate_vector<2, 3, double>{{4.0, 5.0, 6.0}};

template <auto InputSize, auto OutputSize, class RealType>
struct layer_batch_propagate {
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> std::vector<std::array<real_type, OutputSize>> {
    return {output, output};
  }
};
constexpr std::array<std::array<double, 1>, 2> batch_input1_2{input1, input1};
constexpr nou::layer auto layer_input1_output2_batch_propagate =
    layer_batch_propagate<1, 2, double>{{2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_batch_propagate =
    layer_batch_propagate<2, 3, double>{{4.0, 5.0, 6.0}};

template <auto InputSize, auto OutputSize, class RealType>
struct layer_batch_propagate_error {
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> std::expected<std::vector<std::array<real_type, OutputSize>>,
                       nou::computation_error> {
    return std::unexpected{nou::computation_error::invalid_input};
  }
};
constexpr nou::layer auto layer_input1_output2_batch_propagate_error =
    layer_batch_propagate_error<1, 2, double>{};
constexpr nou::layer auto layer_input2_output3_batch_propagate_error =
    layer_batch_propagate_error<2, 3, double>{};

template <auto InputSize, auto OutputSize, class RealType>
struct layer_batch_propagate_errorable {
  using real_type = RealType;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> std::expected<std::vector<std::array<real_type, OutputSize>>,
                       nou::computation_error> {
    return std::expected<std::vector<std::array<real_type, OutputSize>>,
                         nou::computation_error>{{output, output}};
  }
};
constexpr nou::layer auto layer_input1_output2_batch_propagate_errorable =
    layer_batch_propagate_errorable<1, 2, double>{{2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_batch_propagate_errorable =
    layer_batch_propagate_errorable<2, 3, double>{{4.0, 5.0, 6.0}};

template <auto InputSize, auto OutputSize>
struct layer_propagate_noexcept {
  using real_type = double;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)

  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input)
      const noexcept -> std::array<real_type, OutputSize> {
    return output;
  }
};
constexpr nou::layer auto layer_input1_output2_propagate_noexcept =
    layer_propagate_noexcept<1, 2>{{2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_propagate_noexcept =
    layer_propagate_noexcept<2, 3>{{4.0, 5.0, 6.0}};

template <auto InputSize, auto OutputSize>
struct layer_propagate_may_throw {
  using real_type = double;
  static constexpr auto input_extents =
      std::extents<std::size_t, static_cast<std::size_t>(InputSize)>{};
  static constexpr auto output_extents =
      std::extents<std::size_t, static_cast<std::size_t>(OutputSize)>{};
  std::array<real_type, OutputSize>
      output{};  // NOLINT(misc-non-private-member-variables-in-classes)
  template <std::ranges::input_range Input>
  [[nodiscard]] constexpr auto propagate([[maybe_unused]] Input&& input) const
      -> std::array<real_type, OutputSize> {
    return output;
  }
};
constexpr nou::layer auto layer_input1_output2_propagate_may_throw =
    layer_propagate_may_throw<1, 2>{{2.0, 3.0}};
constexpr nou::layer auto layer_input2_output3_propagate_may_throw =
    layer_propagate_may_throw<2, 3>{{4.0, 5.0, 6.0}};
}  // namespace

auto main() -> int {
  // --- prediction propagates its input to the first layer ---
  {
    constexpr auto passed_input_to_first_layer = []() -> auto {
      nou::sequential_network auto network =
          nou::make_network(layer_input1_output2_propagate_member,
                            layer_input2_output3_propagate_member);
      nou::predict(network, input1);
      return std::get<0>(network).passed_input;
    }();
    static_assert(passed_input_to_first_layer == input1);
  }

  // --- prediction propagates layer outputs to subsequent layers ---
  {
    constexpr auto passed_input_to_second_layer = []() -> auto {
      nou::sequential_network auto network =
          nou::make_network(layer_input1_output2_propagate_member,
                            layer_input2_output3_propagate_member);
      nou::predict(network, input1);
      return std::get<1>(network).passed_input;
    }();
    static_assert(passed_input_to_second_layer ==
                  layer_input1_output2_propagate_member.output);
  }

  // --- prediction returns the output of the last layer ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_member);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_member.output);
  }

  // --- predict network via propagate member function ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_member);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_member.output);
  }

  // --- predict network via propagate free function ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_free_func,
                          layer_input2_output3_propagate_free_func);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_free_func.output);
  }

  // --- predict network via propagate hidden friend function ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_hidden_friend,
                          layer_input2_output3_propagate_hidden_friend);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_hidden_friend.output);
  }

  // --- predict const lvalue network ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_member);
    static_assert(nou::predict(network, input1) ==
                  layer_input2_output3_propagate_member.output);
  }

  // --- predict rvalue network ---
  {
    constexpr auto output =
        nou::predict(nou::make_network(layer_input1_output2_propagate_member,
                                       layer_input2_output3_propagate_member),
                     input1);
    static_assert(output == layer_input2_output3_propagate_member.output);
  }

  // --- prediction fails if layer cannot propagate ---
  {
    static_assert(
        compile_fail_predict<decltype(nou::make_network(
                                 not_propagatable_layer_input1_output2,
                                 layer_input2_output3_propagate_member)),
                             decltype(input1)>);
  }

  // --- prediction returns error if the layer returns error ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_with_error,
                          layer_input2_output3_propagate_member),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- prediction returns errorable output if the layer returns
  //     errorable output ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          errorable_layer_input2_output3_without_error),
        input1);
    static_assert(result.value() ==
                  errorable_layer_input2_output3_without_error.output);
  }

  // --- predict [errorable layer(without error),
  //              non-errorable layer] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          layer_input2_output3_propagate_member),
        input1);
    static_assert(result.value() ==
                  layer_input2_output3_propagate_member.output);
  }

  // --- predict [errorable layer(without error),
  //              errorable layer(without error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          errorable_layer_input2_output3_without_error),
        input1);
    static_assert(result.value() ==
                  errorable_layer_input2_output3_without_error.output);
  }

  // --- predict [errorable layer(without error),
  //              errorable layer(with error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          errorable_layer_input2_output3_with_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable layer(with error),
  //              non-errorable layer] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_with_error,
                          layer_input2_output3_propagate_member),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable layer(with error),
  //              errorable layer(without error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_with_error,
                          errorable_layer_input2_output3_without_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable layer(with error),
  //              errorable layer(with error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_with_error,
                          errorable_layer_input2_output3_with_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable layer(without error), non-errorable layer,
  //              errorable layer(without error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          layer_input2_output3_propagate_member,
                          errorable_layer_input3_output4_without_error),
        input1);
    static_assert(result.value() ==
                  errorable_layer_input3_output4_without_error.output);
  }

  // --- predict [errorable layer(without error), non-errorable layer,
  //              errorable layer(with error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          layer_input2_output3_propagate_member,
                          errorable_layer_input3_output4_with_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable layer(with error), non-errorable layer,
  //              errorable layer(without error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_with_error,
                          layer_input2_output3_propagate_member,
                          errorable_layer_input3_output4_without_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable layer(with error), non-errorable layer,
  //              errorable layer(with error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_with_error,
                          layer_input2_output3_propagate_member,
                          errorable_layer_input3_output4_with_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [non-errorable, errorable(without error), non-errorable] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(layer_input1_output2_propagate_member,
                          errorable_layer_input2_output3_without_error,
                          layer_input3_output4_propagate_member),
        input1);
    static_assert(result.value() ==
                  layer_input3_output4_propagate_member.output);
  }

  // --- predict [non-errorable, errorable(with error), non-errorable] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(layer_input1_output2_propagate_member,
                          errorable_layer_input2_output3_with_error,
                          layer_input3_output4_propagate_member),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable(without error), errorable(without error),
  //              errorable(without error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          errorable_layer_input2_output3_without_error,
                          errorable_layer_input3_output4_without_error),
        input1);
    static_assert(result.value() ==
                  errorable_layer_input3_output4_without_error.output);
  }

  // --- predict [errorable(with error), errorable(without error),
  //              errorable(without error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_with_error,
                          errorable_layer_input2_output3_without_error,
                          errorable_layer_input3_output4_without_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable(without error), errorable(with error),
  //              errorable(without error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          errorable_layer_input2_output3_with_error,
                          errorable_layer_input3_output4_without_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict [errorable(without error), errorable(without error),
  //              errorable(with error)] ---
  {
    constexpr auto result = nou::predict(
        nou::make_network(errorable_layer_input1_output2_without_error,
                          errorable_layer_input2_output3_without_error,
                          errorable_layer_input3_output4_with_error),
        input1);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- predict layers propagating from std::vector to std::vector
  //     (input is std::array) ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_vector,
                          layer_input2_output3_propagate_vector);
    static_assert(
        std::ranges::equal(nou::predict(network, input1),
                           layer_input2_output3_propagate_vector.output));
  }

  // --- predict layers propagating from std::array to std::vector
  //     (input is std::array) ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_vector);
    static_assert(
        std::ranges::equal(nou::predict(network, input1),
                           layer_input2_output3_propagate_vector.output));
  }

  // --- predict layers propagating from std::vector to std::vector
  //     (input is std::vector) ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_vector,
                          layer_input2_output3_propagate_vector);
    static_assert(std::ranges::equal(
        nou::predict(network, std::ranges::to<std::vector<double>>(input1)),
        layer_input2_output3_propagate_vector.output));
  }

  // --- predict layers propagating from std::array to std::vector
  //     (input is std::vector) ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_vector);
    static_assert(std::ranges::equal(
        nou::predict(network, std::ranges::to<std::vector<double>>(input1)),
        layer_input2_output3_propagate_vector.output));
  }

  // --- prediction fails if input is array and extents do not match ---
  {
    static_assert(
        compile_fail_predict<decltype(nou::make_network(
                                 layer_input1_output2_propagate_member,
                                 layer_input2_output3_propagate_member)),
                             std::array<double, 2>>);
  }

  // --- prediction success if input extents are dynamic ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_member,
                          layer_input2_output3_propagate_member);
    static_assert(nou::predict(network, std::vector<double>(1.0, 1.0)) ==
                  layer_input2_output3_propagate_member.output);
  }

  // --- prediction fails if input type do not match ---
  {
    static_assert(
        compile_fail_predict<decltype(nou::make_network(
                                 layer_input1_output2_propagate_member,
                                 layer_input2_output3_propagate_member)),
                             std::array<int, 1>>);
  }

  // --- batch prediction ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_batch_propagate,
                          layer_input2_output3_batch_propagate);
    static_assert(std::ranges::equal(
        nou::predict(network, batch_input1_2),
        std::vector{layer_input2_output3_batch_propagate.output,
                    layer_input2_output3_batch_propagate.output}));
  }

  // --- batch prediction with errorable output with error ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_batch_propagate_error,
                          layer_input2_output3_batch_propagate_error);
    constexpr auto result = nou::predict(network, batch_input1_2);
    static_assert(result.error() == nou::computation_error::invalid_input);
  }

  // --- batch prediction with errorable output without error ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_batch_propagate_errorable,
                          layer_input2_output3_batch_propagate_errorable);
    static_assert(std::ranges::equal(
        nou::predict(network, batch_input1_2).value(),
        std::vector{layer_input2_output3_batch_propagate_errorable.output,
                    layer_input2_output3_batch_propagate_errorable.output}));
  }

  // --- noexcept propagation ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_noexcept,
                          layer_input2_output3_propagate_noexcept);
    static_assert(noexcept(nou::predict(network, input1)));
  }

  // --- may-throw propagation ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_may_throw,
                          layer_input2_output3_propagate_may_throw);
    static_assert(not noexcept(nou::predict(network, input1)));
  }

  // --- mixed noexcept and may-throw propagation ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_noexcept,
                          layer_input2_output3_propagate_may_throw);
    static_assert(not noexcept(nou::predict(network, input1)));
  }

  // --- mixed may-throw and noexcept propagation ---
  {
    constexpr nou::sequential_network auto network =
        nou::make_network(layer_input1_output2_propagate_may_throw,
                          layer_input2_output3_propagate_noexcept);
    static_assert(not noexcept(nou::predict(network, input1)));
  }

  return 0;
}