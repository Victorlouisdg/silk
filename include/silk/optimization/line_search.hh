#include <Eigen/Core>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>
#include <TinyAD/Utils/Out.hh>

namespace silk {

template<typename PassiveT, int d>
bool armijo_condition(const PassiveT _f_curr,
                      const PassiveT newValue,
                      const PassiveT _s,
                      const Eigen::Vector<PassiveT, d> &searchDirection,
                      const Eigen::Vector<PassiveT, d> &gradient,
                      const PassiveT _armijo_const) {
  return newValue <= _f_curr + _armijo_const * _s * searchDirection.dot(gradient);
}

template<typename PassiveT, int d, typename EvalFunctionT>
Eigen::Vector<PassiveT, d> lineSearch(
    const Eigen::Vector<PassiveT, d> &initialInputs,
    const Eigen::Vector<PassiveT, d> &searchDirection,
    const PassiveT initialValue,
    const Eigen::Vector<PassiveT, d> &gradient,
    const EvalFunctionT &objectiveFunction,  // Callable of type T(const Eigen::Vector<T, d>&)
    const PassiveT maxStepSize = 1.0,        // Initial step size
    const PassiveT stepSizeShrinkFactor = 0.8,
    const int maxStepSizesToTry = 64,
    const PassiveT _armijo_const = 1e-4) {
  // Check input
  TINYAD_ASSERT_EQ(initialInputs.size(), gradient.size());
  if (maxStepSize <= 0.0)
    TINYAD_ERROR_throw("Max step size not positive.");

  const bool isDescentDirection = gradient.dot(searchDirection) < 0.0;
  std::cout << "isDescentDirection: " << isDescentDirection << std::endl;

  // Also try a step size of 1.0 (if valid)
  const bool tryStepSizeOne = maxStepSize > 1.0;

  Eigen::Vector<PassiveT, d> inputs = initialInputs;
  PassiveT stepSize = maxStepSize;
  for (int i = 0; i < maxStepSizesToTry; ++i) {
    inputs = initialInputs + stepSize * searchDirection;
    const PassiveT newValue = objectiveFunction(inputs);
    if (armijo_condition(initialValue, newValue, stepSize, searchDirection, gradient, _armijo_const)) {
      std::cout << "Line search needed " << i + 1 << " iterations." << std::endl;
      return inputs;
    }

    if (tryStepSizeOne && stepSize > 1.0 && stepSize * stepSizeShrinkFactor < 1.0) {
      stepSize = 1.0;
      continue;
    }

    stepSize *= stepSizeShrinkFactor;
  }

  TINYAD_WARNING("Line search couldn't find improvement. Gradient max norm is " << gradient.cwiseAbs().maxCoeff());
  return initialInputs;
}

}  // namespace silk