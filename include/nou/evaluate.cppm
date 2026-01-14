module;

#include <expected>
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
           mathematical_function<LossFunction,
                                 successful_predict_result_t<Network, Input>,
                                 Target>
// NOLINTNEXTLINE(misc-use-internal-linkage)
constexpr auto evaluate(Network&& network, Input&& input, Target&& target,
                        LossFunction loss_function) -> auto {
  using successful_output_type = decltype(loss_function(
      std::declval<successful_predict_result_t<Network, Input>>(),
      std::forward<Target>(target)));
  auto output =
      predict(std::forward<Network>(network), std::forward<Input>(input));

  if constexpr (computation_errorable<decltype(output)>) {
    if (!output.has_value()) {
      return std::expected<successful_output_type, computation_error>{
          std::unexpect, output.error()};
    }
    return std::expected<successful_output_type, computation_error>{
        loss_function(std::move(output).value(), std::forward<Target>(target))};
  } else {
    return loss_function(std::move(output), std::forward<Target>(target));
  }
}

}  // namespace nou