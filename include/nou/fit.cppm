module;

#include <ranges>
#include <tuple>
#include <type_traits>
#include <utility>

export module nou.fit;

export import nou.evaluate;

namespace nou {

// A single batch pair: (batch_input, batch_target)
template <class T, class Network>
concept single_batch =
    (std::tuple_size_v<std::remove_cvref_t<T>> >= 2) &&
    batch_input<std::tuple_element_t<0, std::remove_cvref_t<T>>,
                first_layer_t<Network>> &&
    batch_target<std::tuple_element_t<1, std::remove_cvref_t<T>>, Network,
                 std::tuple_element_t<0, std::remove_cvref_t<T>>>;

// A batch dataset is a range of single_batch items
template <class T, class Network>
concept batch_dataset =
    std::ranges::input_range<T> &&
    single_batch<std::ranges::range_value_t<std::remove_cvref_t<T>>, Network>;

// An epoch dataset is a range of batch datasets
template <class T, class Network>
concept epoch_dataset =
    std::ranges::input_range<T> &&
    batch_dataset<std::ranges::range_value_t<std::remove_cvref_t<T>>, Network>;

// Accessors for a single_batch pair
template <class SingleBatch>
using batch_input_t = std::tuple_element_t<0, std::remove_cvref_t<SingleBatch>>;

template <class SingleBatch>
using batch_target_t =
    std::tuple_element_t<1, std::remove_cvref_t<SingleBatch>>;

// The loss function must be callable either on whole-batch outputs/targets
// or element-wise on per-sample output/target.
template <class T, class Network, class SingleBatch>
concept callable_loss_function =
    single_batch<SingleBatch, Network> &&
    (differentiable_function<
         T, successful_predict_result_t<Network, batch_input_t<SingleBatch>>,
         batch_target_t<SingleBatch>> ||
     differentiable_function<
         T,
         std::ranges::range_value_t<
             successful_predict_result_t<Network, batch_input_t<SingleBatch>>>,
         std::ranges::range_value_t<batch_target_t<SingleBatch>>>);

// Core: compute one batch loss from a single_batch pair
template <class Network, class SingleBatch, class Optimizers,
          class LossFunction>
  requires sequential_network<std::remove_cvref_t<Network>> &&
           single_batch<SingleBatch, Network> &&
           callable_loss_function<LossFunction, Network, SingleBatch>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto fit_impl(Network&& network, SingleBatch&& dataset,
                        [[maybe_unused]] Optimizers&& optimizers,
                        LossFunction loss_function) -> auto {
  auto&& [batch_input, batch_target] = std::forward<SingleBatch>(dataset);
  // 2. Propagate the input through the network
  auto output = predict(std::forward<Network>(network),
                        std::forward_like<SingleBatch>(batch_input));
  // 3. Compute the loss using the loss function
  auto loss = loss_function(std::move(output),
                            std::forward_like<SingleBatch>(batch_target));
  // 4. Compute gradients using back-propagation
  // 5. Update network parameters using the optimizers
  // 6. Return the computed loss value obtained at step 3
  return loss;
}

template <class Network, class BatchDataset, class Optimizers,
          class LossFunction>
  requires sequential_network<std::remove_cvref_t<Network>> &&
           batch_dataset<std::remove_cvref_t<BatchDataset>, Network> &&
           (!single_batch<std::remove_cvref_t<BatchDataset>, Network>)
// NOLINTNEXTLINE
constexpr auto fit_impl(Network&& network, BatchDataset&& dataset,
                        // NOLINTNEXTLINE
                        Optimizers&& optimizers, LossFunction loss_function)
    -> auto {
  using result_type = decltype(fit_impl(
      std::forward<Network>(network),
      *std::ranges::begin(std::forward<BatchDataset>(dataset)),
      std::forward<Optimizers>(optimizers), loss_function));

  result_type total_loss{0};
  std::size_t count = 0;

  for (auto&& batch : dataset) {
    total_loss += fit_impl(network, batch, optimizers, loss_function);
    count++;
  }

  if (count == 0) {
    return result_type{0};
  }
  return total_loss / static_cast<result_type>(count);
}

export template <class Network, class Optimizers, class LossFunction>
  requires sequential_network<std::remove_cvref_t<Network>>
constexpr auto fit(Network&& network, Optimizers&& optimizers,
                   LossFunction loss_function) -> auto {
  return std::views::transform(
      [&, loss_function = std::move(loss_function)](auto&& elem) -> auto {
        return fit_impl(network, std::forward<decltype(elem)>(elem), optimizers,
                        loss_function);
      });
}

export template <class Network, class Dataset, class Optimizers,
                 class LossFunction>
  requires sequential_network<std::remove_cvref_t<Network>> &&
           (batch_dataset<Dataset, Network> || epoch_dataset<Dataset, Network>)
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto fit(Network&& network, Dataset&& dataset,
                   Optimizers&& optimizers, LossFunction loss_function)
    -> auto {
  return std::forward<Dataset>(dataset) |
         fit(std::forward<Network>(network),
             std::forward<Optimizers>(optimizers), std::move(loss_function));
}

}  // namespace nou
