module;

#include <concepts>
#include <utility>

export module nou.evaluate;

export import nou.predict;

namespace nou {

export template <class Network, input<first_layer_t<Network>> Input,
                 class Target, class LossFunction>
  requires compatible_extents_with<
               Target, typename last_layer_t<Network>::real_type,
               decltype(auto(last_layer_t<Network>::output_extents))> &&
           requires {
             {
               predict(std::declval<Network>(), std::declval<Input>())
             } -> output<last_layer_t<Network>>;
           } &&
           std::invocable<LossFunction,
                          successful_predict_result_t<Network, Input>, Target>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, Input&& input, Target&& target,
                        LossFunction loss_function) -> auto {
  auto output =
      predict(std::forward<Network>(network), std::forward<Input>(input));
  return loss_function(std::move(output), std::forward<Target>(target));
}  // namespace nou

}  // namespace nou