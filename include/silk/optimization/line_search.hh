#include <Eigen/Core>
#include <TinyAD/Detail/EigenVectorTypedefs.hh>
#include <TinyAD/Utils/Out.hh>

namespace silk {

bool convergenceConditionIPC(const Eigen::VectorXd &searchDirection,
                             double timestepSize,
                             double convergenceAccuracy = 1e-5) {
  // Infinity norm of search direction, see the "Termination" paragraph in the IPC paper.
  double searchMax = searchDirection.cwiseAbs().maxCoeff();
  double stoppingMeasure = searchMax / timestepSize;
  return stoppingMeasure < convergenceAccuracy;
}

template<typename T, int d>
bool armijoCondition(const T initialValue,
                     const T newValue,
                     const T stepSize,
                     const Eigen::Vector<T, d> &searchDirection,
                     const Eigen::Vector<T, d> &gradient,
                     const T armijoConstant) {
  // std::cout << "armijoCondition: " << newValue << " <= " << initialValue << " + " << armijoConstant << " * "
  //           << stepSize << " * " << searchDirection.dot(gradient) << std::endl;
  return newValue <= initialValue + armijoConstant * stepSize * searchDirection.dot(gradient);  // Equation 3.4
}

template<typename T, int d, typename EvalFunctionT>
Eigen::Vector<T, d> backtrackingLineSearch(
    const Eigen::Vector<T, d> &initialVariables,
    const Eigen::Vector<T, d> &searchDirection,
    const T initialValue,
    const Eigen::Vector<T, d> &gradient,
    const EvalFunctionT &objectiveFunction,  // Callable of type T(const Eigen::Vector<T, d>&)
    const T maxStepSize = 1.0,               // Initial step size
    const T contractionFactor = 0.8,
    const int maxStepSizesToTry = 64,
    const T armijoConstant = 1e-4) {
  // Check input
  TINYAD_ASSERT_EQ(initialVariables.size(), gradient.size());
  if (maxStepSize <= 0.0)
    TINYAD_ERROR_throw("Max step size not positive.");

  // const bool isDescentDirection = gradient.dot(searchDirection) < 0.0;
  // std::cout << "isDescentDirection: " << isDescentDirection << std::endl;

  // Also try a step size of 1.0 (if valid)
  const bool tryStepSizeOne = maxStepSize > 1.0;

  Eigen::Vector<T, d> variables = initialVariables;
  T stepSize = maxStepSize;
  for (int i = 0; i < maxStepSizesToTry; ++i) {
    variables = initialVariables + stepSize * searchDirection;  // Equation 3.1 in Numerical Optimization
    const T newValue = objectiveFunction(variables);
    if (armijoCondition(initialValue, newValue, stepSize, searchDirection, gradient, armijoConstant)) {
      // std::cout << "Line search needed " << i + 1 << " iterations." << std::endl;
      return variables;
    }

    if (tryStepSizeOne && stepSize > 1.0 && stepSize * contractionFactor < 1.0) {
      stepSize = 1.0;
      continue;
    }

    stepSize *= contractionFactor;
  }

  TINYAD_WARNING("Line search couldn't find improvement. Gradient max norm is " << gradient.cwiseAbs().maxCoeff());
  return initialVariables;
}

}  // namespace silk