module;

#include <utility>

export module nou.evaluate;

export import nou.predict;

namespace nou {

export template <class Network, class Input, class Target, class LossFunction>
constexpr auto evaluate(Network&& network, Input&& input, Target&& target,
                        LossFunction loss_function) -> auto {
  auto output =
      predict(std::forward<Network>(network), std::forward<Input>(input));
  return loss_function(std::move(output), std::forward<Target>(target));
}

}  // namespace nou