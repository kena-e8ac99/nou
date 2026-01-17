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

// Internal helper: Calls the loss function(s)/metric(s) with the given output
// and target.
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

    auto total_loss = []() -> value_type {
      if constexpr (std::is_arithmetic_v<value_type>) {
        return value_type{0};
      } else {
        return value_type{};
      }
    }();

    auto size = std::ranges::size(output);
    if (size == 0) {
      return total_loss;
    }

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
            std::ranges::end(output), std::ranges::begin(target), total_loss,
            std::plus<>{}, [&](auto&& output, auto&& target) -> value_type {
              return loss_function(std::forward<decltype(output)>(output),
                                   std::forward<decltype(target)>(target));
            });
        return total_loss / static_cast<value_type>(size);
      }
    }

    total_loss = std::transform_reduce(
        std::ranges::begin(output), std::ranges::end(output),
        std::ranges::begin(target), total_loss, std::plus<>{},
        [&](auto&& output, auto&& target) -> value_type {
          return loss_function(std::forward<decltype(output)>(output),
                               std::forward<decltype(target)>(target));
        });
    return total_loss / static_cast<value_type>(size);
  }
}

// Internal helper: Variadic call to multiple loss functions / metrics.
template <class Output, class Target, class... Functions>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto call_loss_function(Output&& output, Target&& target,
                                  Functions... functions) -> auto {
  if constexpr (sizeof...(functions) == 1) {
    return call_loss_function(std::forward<Output>(output),
                              std::forward<Target>(target),
                              std::move(functions)...);
  } else {
    return std::tuple{
        call_loss_function(output, target, std::move(functions))...};
  }
}

// Internal helper: Implementation of the evaluate function template.
template <class Network, class Input, class Target, class... Functions>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate_impl(Network&& network, Input&& input, Target&& target,
                             Functions... functions) -> auto {
  using successful_output_type = decltype(call_loss_function(
      std::declval<successful_predict_result_t<Network, Input>>(),
      std::forward<Target>(target), std::move(functions)...));
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
                           std::move(functions)...)};
  } else {
    return call_loss_function(std::move(output), std::forward<Target>(target),
                              std::move(functions)...);
  }
}

/**
 * @brief Evaluates a neural network with a single input using a differentiable
 * loss function and optional metrics.
 *
 * This function performs forward propagation of a single input through the
 * sequential network and computes a differentiable loss function along with
 * optional evaluation metrics by comparing the network's output against the
 * target.
 *
 * ### Error handling
 * If the network's `predict` function produces an errorable output (satisfying
 * `computation_errorable`), the output of `evaluate` will also be errorable
 * (wrapped in `std::expected`). If an intermediate computation returns a
 * `computation_error`, the process is short-circuited and the error is returned
 * immediately.
 *
 * ### Example
 * ```cpp
 * // Single metric case
 * auto loss = nou::evaluate(network, input, target, loss_function, metric);
 *
 * // Multiple metrics case (returns tuple of loss and metrics)
 * auto [loss, accuracy, precision] =
 *     nou::evaluate(network, input, target, loss_function, accuracy_metric,
 *                   precision_metric);
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model
 * `sequential_network`.
 * @tparam Input The type of the single input. This type must model
 * `input<first_layer_t<Network>>`.
 * @tparam Target The type of the target output. This type must satisfy
 * `compatible_extents_with<Target, last_layer_t<Network>::real_type,
 * last_layer_t<Network>::output_extents>`.
 * @tparam LossFunction The type of the loss function. This type must model
 * `differentiable_function<LossFunction, Output, Target>` where `Output` is
 * the successful prediction result type.
 * @tparam Metrics The types of the metric functions. Each type must model
 * `mathematical_function<Metric, Output, Target>` where `Output` is the
 * successful prediction result type.
 * @param network The sequential network instance.
 * @param input The single input to propagate through the network.
 * @param target The target output for comparison.
 * @param loss_function The differentiable loss function to compute.
 * @param metrics Optional metric functions to compute alongside the loss.
 * @return If no metrics are provided, returns the loss value. If one or more
 * metrics are provided, returns a tuple containing the loss value followed by
 * the metric values. The return type may be wrapped in `std::expected` if the
 * network produces errorable outputs.
 *
 * @see sequential_network for the requirements on the Network type.
 * @see input for the requirements on the Input type.
 * @see differentiable_function for the requirements on the LossFunction type.
 * @see mathematical_function for the requirements on the Metrics types.
 * @see predict for the underlying network propagation mechanism.
 */
export template <class Network, class Input, class Target, class LossFunction,
                 class... Metrics>
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
           differentiable_function<LossFunction,
                                   successful_predict_result_t<Network, Input>,
                                   Target> &&
           (mathematical_function<
                Metrics, successful_predict_result_t<Network, Input>, Target> &&
            ...)
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, Input&& input, Target&& target,
                        LossFunction loss_function, Metrics... metrics)
    -> auto {
  return evaluate_impl(std::forward<Network>(network),
                       std::forward<Input>(input), std::forward<Target>(target),
                       std::move(loss_function), std::move(metrics)...);
}

/**
 * @brief Evaluates a neural network with a single input using metric functions
 * only.
 *
 * This function performs forward propagation of a single input through the
 * sequential network and computes evaluation metrics by comparing the network's
 * output against the target. This overload is selected when no differentiable
 * loss function is provided.
 *
 * ### Error handling
 * If the network's `predict` function produces an errorable output (satisfying
 * `computation_errorable`), the output of `evaluate` will also be errorable
 * (wrapped in `std::expected`). If an intermediate computation returns a
 * `computation_error`, the process is short-circuited and the error is returned
 * immediately.
 *
 * ### Example
 * ```cpp
 * // Single metric case
 * auto accuracy = nou::evaluate(network, input, target, accuracy_metric);
 *
 * // Multiple metrics case (returns tuple)
 * auto [accuracy, precision] =
 *     nou::evaluate(network, input, target, accuracy_metric, precision_metric);
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model
 * `sequential_network`.
 * @tparam Input The type of the single input. This type must model
 * `input<first_layer_t<Network>>`.
 * @tparam Target The type of the target output. This type must satisfy
 * `compatible_extents_with<Target, last_layer_t<Network>::real_type,
 * last_layer_t<Network>::output_extents>`.
 * @tparam Metrics The types of the metric functions. Each type must model
 * `mathematical_function<Metric, Output, Target>` where `Output` is the
 * successful prediction result type. None of the metric functions can model
 * `differentiable_function` (otherwise the overload with a loss function would
 * be selected).
 * @param network The sequential network instance.
 * @param input The single input to propagate through the network.
 * @param target The target output for comparison.
 * @param metrics One or more metric functions to compute.
 * @return If one metric is provided, returns the metric value. If multiple
 * metrics are provided, returns a tuple containing the metric values. The
 * return type may be wrapped in `std::expected` if the network produces
 * errorable outputs.
 *
 * @see sequential_network for the requirements on the Network type.
 * @see input for the requirements on the Input type.
 * @see mathematical_function for the requirements on the Metrics types.
 * @see predict for the underlying network propagation mechanism.
 */
export template <class Network, class Input, class Target, class... Metrics>
  requires sequential_network<std::remove_cvref_t<Network>> &&
           input<Input, first_layer_t<Network>> &&
           compatible_extents_with<
               Target, typename last_layer_t<Network>::real_type,
               decltype(auto(last_layer_t<Network>::output_extents))> &&
           requires {
             {
               predict(std::declval<Network>(), std::declval<Input>())
             } -> output<last_layer_t<Network>>;
           } && (sizeof...(Metrics) >= 1) &&
           (mathematical_function<
                Metrics, successful_predict_result_t<Network, Input>, Target> &&
            ...) &&
           (!differentiable_function<
                Metrics, successful_predict_result_t<Network, Input>, Target> &&
            ...)
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, Input&& input, Target&& target,
                        Metrics... metrics) -> auto {
  return evaluate_impl(std::forward<Network>(network),
                       std::forward<Input>(input), std::forward<Target>(target),
                       std::move(metrics)...);
}

/**
 * @brief Evaluates a neural network with batch input using a differentiable
 * loss function and optional metrics.
 *
 * This function performs forward propagation of multiple samples (batch input)
 * through the sequential network and computes a differentiable loss function
 * along with optional evaluation metrics by comparing the network's batch
 * output against the batch target. The loss function and metrics can operate
 * either on the entire batch or element-wise on individual samples.
 *
 * ### Batch vs element-wise operation
 * - If the loss function accepts the entire batch output and batch target, it
 *   is called once with the full batches
 * - If the loss function accepts individual samples, it is applied to each
 *   sample pair and the results are averaged. For `const` loss functions,
 *   element-wise computation uses parallel execution
 * (`std::execution::par_unseq`) when not in a constant evaluation context
 *
 * ### Error handling
 * If the network's `predict` function produces an errorable output (satisfying
 * `computation_errorable`), the output of `evaluate` will also be errorable
 * (wrapped in `std::expected`). If an intermediate computation returns a
 * `computation_error`, the process is short-circuited and the error is returned
 * immediately.
 *
 * ### Example
 * ```cpp
 * // Batch evaluation with loss and metrics
 * auto batch_inputs = std::vector<std::array<float, 3>>{{...}, {...}};
 * auto batch_targets = std::vector<std::array<float, 2>>{{...}, {...}};
 * auto [loss, accuracy] =
 *     nou::evaluate(network, batch_inputs, batch_targets,
 *                   loss_function, accuracy_metric);
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model
 * `sequential_network`.
 * @tparam BatchInput The type of the batch input. This type must model
 * `batch_input<first_layer_t<Network>>`.
 * @tparam BatchTarget The type of the batch target. This type must model
 * `std::ranges::input_range` where each element satisfies
 * `compatible_extents_with<Element, last_layer_t<Network>::real_type,
 * last_layer_t<Network>::output_extents>`.
 * @tparam LossFunction The type of the loss function. This type must model
 * either `differentiable_function<LossFunction, BatchOutput, BatchTarget>` or
 * `differentiable_function<LossFunction, OutputElement, TargetElement>` where
 * `BatchOutput` is the successful batch prediction result type, and
 * `OutputElement` and `TargetElement` are the individual sample types.
 * @tparam Metrics The types of the metric functions. Each type must model
 * either `mathematical_function<Metric, BatchOutput, BatchTarget>` or
 * `mathematical_function<Metric, OutputElement, TargetElement>`.
 * @param network The sequential network instance.
 * @param input The batch input to propagate through the network.
 * @param target The batch target for comparison.
 * @param loss_function The differentiable loss function to compute.
 * @param metrics Optional metric functions to compute alongside the loss.
 * @return If no metrics are provided, returns the loss value. If one or more
 * metrics are provided, returns a tuple containing the loss value followed by
 * the metric values. The return type may be wrapped in `std::expected` if the
 * network produces errorable outputs.
 *
 * @see sequential_network for the requirements on the Network type.
 * @see batch_input for the requirements on the BatchInput type.
 * @see differentiable_function for the requirements on the LossFunction type.
 * @see mathematical_function for the requirements on the Metrics types.
 * @see predict for the underlying network propagation mechanism.
 */
export template <class Network, class BatchInput, class BatchTarget,
                 class LossFunction, class... Metrics>
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
           (differentiable_function<
                LossFunction, successful_predict_result_t<Network, BatchInput>,
                BatchTarget> ||
            differentiable_function<
                LossFunction,
                std::ranges::range_value_t<
                    successful_predict_result_t<Network, BatchInput>>,
                std::ranges::range_value_t<
                    std::remove_cvref_t<BatchTarget>>>) &&
           ((mathematical_function<
                 Metrics, successful_predict_result_t<Network, BatchInput>,
                 BatchTarget> ||
             mathematical_function<
                 Metrics,
                 std::ranges::range_value_t<
                     successful_predict_result_t<Network, BatchInput>>,
                 std::ranges::range_value_t<
                     std::remove_cvref_t<BatchTarget>>>) &&
            ...)
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, BatchInput&& input,
                        BatchTarget&& target, LossFunction loss_function,
                        Metrics... metrics) -> auto {
  return evaluate_impl(std::forward<Network>(network),
                       std::forward<BatchInput>(input),
                       std::forward<BatchTarget>(target),
                       std::move(loss_function), std::move(metrics)...);
}

/**
 * @brief Evaluates a neural network with batch input using metric functions
 * only.
 *
 * This function performs forward propagation of multiple samples (batch input)
 * through the sequential network and computes evaluation metrics by comparing
 * the network's batch output against the batch target. This overload is
 * selected when no differentiable loss function is provided. The metrics can
 * operate either on the entire batch or element-wise on individual samples.
 *
 * ### Batch vs element-wise operation
 * - If a metric accepts the entire batch output and batch target, it is called
 *   once with the full batches
 * - If a metric accepts individual samples, it is applied to each sample pair
 *   and the results are averaged. For `const` metrics, element-wise computation
 *   uses parallel execution (`std::execution::par_unseq`) when not in a
 *   constant evaluation context
 *
 * ### Error handling
 * If the network's `predict` function produces an errorable output (satisfying
 * `computation_errorable`), the output of `evaluate` will also be errorable
 * (wrapped in `std::expected`). If an intermediate computation returns a
 * `computation_error`, the process is short-circuited and the error is returned
 * immediately.
 *
 * ### Example
 * ```cpp
 * // Batch evaluation with metrics only
 * auto batch_inputs = std::vector<std::array<float, 3>>{{...}, {...}};
 * auto batch_targets = std::vector<std::array<float, 2>>{{...}, {...}};
 * auto [accuracy, precision] =
 *     nou::evaluate(network, batch_inputs, batch_targets,
 *                   accuracy_metric, precision_metric);
 * ```
 *
 * @tparam Network The type of the sequential network. This type must model
 * `sequential_network`.
 * @tparam BatchInput The type of the batch input. This type must model
 * `batch_input<first_layer_t<Network>>`.
 * @tparam BatchTarget The type of the batch target. This type must model
 * `std::ranges::input_range` where each element satisfies
 * `compatible_extents_with<Element, last_layer_t<Network>::real_type,
 * last_layer_t<Network>::output_extents>`.
 * @tparam Metrics The types of the metric functions. Each type must model
 * either `mathematical_function<Metric, BatchOutput, BatchTarget>` or
 * `mathematical_function<Metric, OutputElement, TargetElement>` where
 * `BatchOutput` is the successful batch prediction result type, and
 * `OutputElement` and `TargetElement` are the individual sample types. None of
 * the metric functions can model `differentiable_function` (otherwise the
 * overload with a loss function would be selected).
 * @param network The sequential network instance.
 * @param input The batch input to propagate through the network.
 * @param target The batch target for comparison.
 * @param metrics One or more metric functions to compute.
 * @return If one metric is provided, returns the metric value. If multiple
 * metrics are provided, returns a tuple containing the metric values. The
 * return type may be wrapped in `std::expected` if the network produces
 * errorable outputs.
 *
 * @see sequential_network for the requirements on the Network type.
 * @see batch_input for the requirements on the BatchInput type.
 * @see mathematical_function for the requirements on the Metrics types.
 * @see predict for the underlying network propagation mechanism.
 */
export template <class Network, class BatchInput, class BatchTarget,
                 class... Metrics>
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
           } && (sizeof...(Metrics) >= 1) &&
           ((mathematical_function<
                 Metrics, successful_predict_result_t<Network, BatchInput>,
                 BatchTarget> ||
             mathematical_function<
                 Metrics,
                 std::ranges::range_value_t<
                     successful_predict_result_t<Network, BatchInput>>,
                 std::ranges::range_value_t<
                     std::remove_cvref_t<BatchTarget>>>) &&
            ...) &&
           (!(differentiable_function<
                  Metrics, successful_predict_result_t<Network, BatchInput>,
                  BatchTarget> ||
              differentiable_function<
                  Metrics,
                  std::ranges::range_value_t<
                      successful_predict_result_t<Network, BatchInput>>,
                  std::ranges::range_value_t<
                      std::remove_cvref_t<BatchTarget>>>) &&
            ...)
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, BatchInput&& input,
                        BatchTarget&& target, Metrics... metrics) -> auto {
  return evaluate_impl(
      std::forward<Network>(network), std::forward<BatchInput>(input),
      std::forward<BatchTarget>(target), std::move(metrics)...);
}

}  // namespace nou