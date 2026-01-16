module;

#include <expected>
#include <utility>

export module nou.evaluate;

export import nou.predict;

namespace nou {

export template <class Network, class Input, class Target, class LossFunction>
  requires input<Input, first_layer_t<Network>> &&
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
  using successful_output_type = decltype(loss_function(
      std::declval<successful_predict_result_t<Network, Input>>(),
      std::forward<Target>(target)));
  auto prediction =
      predict(std::forward<Network>(network), std::forward<Input>(input));

  if constexpr (computation_errorable<decltype(prediction)>) {
    if (!prediction.has_value()) {
      return std::expected<successful_output_type, computation_error>{
          std::unexpect, prediction.error()};
    }
    return std::expected<successful_output_type, computation_error>{
        loss_function(std::move(prediction).value(),
                      std::forward<Target>(target))};
  } else {
    return loss_function(std::move(prediction), std::forward<Target>(target));
  }
}

}  // namespace nou