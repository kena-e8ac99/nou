module;

#include <execution>
#include <expected>
#include <numeric>
#include <ranges>
#include <type_traits>
#include <utility>

export module nou.evaluate;

export import nou.predict;

namespace nou {

template <class Output, class Target, class LossFunction>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto call_loss_function(Output&& output, Target&& target,
                                  LossFunction loss_function) -> auto {
  if constexpr (mathematical_function<LossFunction, Output, Target>) {
    return loss_function(std::forward<Output>(output),
                         std::forward<Target>(target));
  } else {
    using value_type = decltype(loss_function(
        std::ranges::range_value_t<std::remove_cvref_t<Output>>{},
        std::ranges::range_value_t<std::remove_cvref_t<Target>>{}));
    value_type total_loss{};

    if !consteval {
      if constexpr (std::ranges::forward_range<Output> &&
                    std::ranges::forward_range<Target> &&
                    mathematical_function<
                        const LossFunction,
                        std::ranges::range_value_t<std::remove_cvref_t<Output>>,
                        std::ranges::range_value_t<
                            std::remove_cvref_t<Target>>>) {
        total_loss = std::transform_reduce(
            std::execution::par_unseq, std::ranges::begin(output),
            std::ranges::end(output), std::ranges::begin(target), value_type{},
            std::plus<>{}, [&](auto&& output, auto&& target) -> value_type {
              return loss_function(std::forward<decltype(output)>(output),
                                   std::forward<decltype(target)>(target));
            });
        return total_loss / static_cast<value_type>(std::ranges::size(output));
      }
      std::unreachable();
    }

    total_loss = std::transform_reduce(
        std::ranges::begin(output), std::ranges::end(output),
        std::ranges::begin(target), value_type{}, std::plus<>{},
        [&loss_function](auto&& output, auto&& target) -> value_type {
          return loss_function(std::forward<decltype(output)>(output),
                               std::forward<decltype(target)>(target));
        });
    return total_loss / static_cast<value_type>(std::ranges::size(output));
  }
}

template <class Network, class Input, class Target, class LossFunction>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate_impl(Network&& network, Input&& input, Target&& target,
                             LossFunction loss_function) -> auto {
  using successful_output_type = decltype(call_loss_function(
      std::declval<successful_predict_result_t<Network, Input>>(),
      std::forward<Target>(target), loss_function));
  auto output =
      predict(std::forward<Network>(network), std::forward<Input>(input));

  if constexpr (computation_errorable<decltype(output)>) {
    if (!output.has_value()) {
      return std::expected<successful_output_type, computation_error>{
          std::unexpect, output.error()};
    }
    return std::expected<successful_output_type, computation_error>{
        call_loss_function(std::move(output).value(),
                           std::forward<Target>(target),
                           std::move(loss_function))};
  } else {
    return call_loss_function(std::move(output), std::forward<Target>(target),
                              std::move(loss_function));
  }
}

export template <class Network, class Input, class Target, class LossFunction>
  requires sequential_network<std::remove_cvref_t<Network>> &&
           input<Input, first_layer_t<Network>> &&
           compatible_extents_with<
               Target, typename last_layer_t<Network>::real_type,
               decltype(auto(last_layer_t<Network>::output_extents))> &&
           requires {
             {
               predict(std::declval<Network>(), std::declval<Input>())
             } -> output<last_layer_t<Network>>;
           } &&
           mathematical_function<LossFunction,
                                 successful_predict_result_t<Network, Input>,
                                 Target>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, Input&& input, Target&& target,
                        LossFunction loss_function) -> auto {
  return evaluate_impl(std::forward<Network>(network),
                       std::forward<Input>(input), std::forward<Target>(target),
                       std::move(loss_function));
}

export template <class Network, class BatchInput, class BatchTarget,
                 class LossFunction>
  requires sequential_network<std::remove_cvref_t<Network>> &&
           batch_input<BatchInput, first_layer_t<Network>> &&
           std::ranges::input_range<BatchTarget> &&
           compatible_extents_with<
               std::ranges::range_value_t<std::remove_cvref_t<BatchTarget>>,
               typename last_layer_t<Network>::real_type,
               decltype(auto(last_layer_t<Network>::output_extents))> &&
           requires {
             {
               predict(std::declval<Network>(), std::declval<BatchInput>())
             } -> batch_output<last_layer_t<Network>>;
           } &&
           (mathematical_function<
                LossFunction, successful_predict_result_t<Network, BatchInput>,
                BatchTarget> ||
            mathematical_function<
                LossFunction,
                std::ranges::range_value_t<
                    successful_predict_result_t<Network, BatchInput>>,
                std::ranges::range_value_t<std::remove_cvref_t<BatchTarget>>>)
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, BatchInput&& input,
                        BatchTarget&& target, LossFunction loss_function)
    -> auto {
  return evaluate_impl(
      std::forward<Network>(network), std::forward<BatchInput>(input),
      std::forward<BatchTarget>(target), std::move(loss_function));
}

}  // namespace nou