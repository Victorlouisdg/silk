#include <Eigen/Core>
#include <TinyAD/ScalarFunction.hh>
#include <iostream>

template<typename T> T elementEnergy0(Eigen::Vector3<T> x) {
  return x.sum();
}

template<typename T> T elementEnergy1(Eigen::Vector3<T> x) {
  return x.prod();
}

TinyAD::ScalarFunction<3, double, Eigen::Index> createScalarFunction(int vertexCount) {
  auto scalarFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));

  scalarFunction.add_elements<1>(TinyAD::range(vertexCount), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    using ScalarT = TINYAD_SCALAR_TYPE(element);
    int index = element.handle;
    Eigen::Vector3<ScalarT> x = element.variables(index);
    ScalarT energy = elementEnergy0(x);
    return energy;
  });

  return scalarFunction;
}

int main() {
  Eigen::RowVector3d v0(0.0, 0.0, 0.0);
  Eigen::RowVector3d v1(1.0, 1.0, 1.0);
  Eigen::Matrix<double, 2, 3> vertexPositions;
  vertexPositions << v0, v1;
  int vertexCount = vertexPositions.rows();

  Eigen::VectorXd x = vertexPositions.reshaped<Eigen::RowMajor>(vertexPositions.rows() * 3, 1);

  auto scalarFunction0 = createScalarFunction(vertexCount /*, elementEnergy0 */);
  auto e0 = scalarFunction0.eval(x);
  std::cout << e0 << std::endl;

  // I want this function to use elementEnergy1:
  auto scalarFunction1 = createScalarFunction(vertexCount /*, elementEnergy1 */);
}