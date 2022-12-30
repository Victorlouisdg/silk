#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <iostream>

#include <TinyAD/ScalarFunction.hh>

#include "silk/types.hh"

using namespace std;

namespace silk {

template<typename ScalarFunction> EnergyFunction standardizeScalarFunction(const ScalarFunction &scalarFunction) {
  EnergyFunction energyFunction = [&](const Eigen::VectorXd &x) { return scalarFunction.eval_with_hessian_proj(x); };
  return energyFunction;
}

/**
 * @brief Create function "maps" a function that calculates the energy of a vertex over an entire mesh. The result is a
 * function that takes in the flattened position of all vertices and returns the total energy of the mesh, its gradient
 * and Hessian.
 *
 * We use TinyAD to do the mapping and the automatic differentiation. This results in a TinyAD ScalarFunction that we
 * then simplify to just a regular function.
 *
 * @tparam F
 * @param vertexCount the amount of vertices in the entire mesh.
 * @param elementEnergy the function template that calculates the energy of the vertex.
 * @return The ScalarFunction
 */
template<typename FunctionTemplate>
EnergyFunction createVertexEnergyFunction(FunctionTemplate &&vertexEnergy, int vertexCount) {

  auto scalarFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));
  scalarFunction.add_elements<1>(TinyAD::range(vertexCount), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    using ScalarT = TINYAD_SCALAR_TYPE(element);
    int index = element.handle;
    Eigen::Vector3<ScalarT> x = element.variables(index);
    ScalarT energy = elementEnergy(x);
    return energy;
  });

  EnergyFunction energyFunction = standardizeScalarFunction(scalarFunction);
  return energyFunction;
}

template<typename FunctionTemplate>
TinyAD::ScalarFunction<3, double, Eigen::Index> createTriangleScalarFunction(
    FunctionTemplate &&triangleEnergy,
    int vertexCount,
    const Triangles &triangles,
    const vector<Eigen::Matrix2d> &triangleInvertedRestShapes) {

  auto scalarFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));
  scalarFunction.add_elements<3>(TinyAD::range(triangles.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    using ScalarT = TINYAD_SCALAR_TYPE(element);
    int triangleIndex = element.handle;

    Eigen::Vector3<ScalarT> x0 = element.variables(triangles(triangleIndex, 0));
    Eigen::Vector3<ScalarT> x1 = element.variables(triangles(triangleIndex, 1));
    Eigen::Vector3<ScalarT> x2 = element.variables(triangles(triangleIndex, 2));

    Eigen::Matrix2d invertedRestShape = triangleInvertedRestShapes[triangleIndex];
    Eigen::Matrix<ScalarT, 3, 2> F = silk::triangleDeformationGradient(x0, x1, x2, invertedRestShape);

    ScalarT energy = triangleEnergy(F);
    return energy;
  });

  return scalarFunction;
}

TinyAD::ScalarFunction<3, double, Eigen::Index> createKineticPotentialFunction(
    const Eigen::VectorXd &positionsAtStartOfTimestep,
    const Eigen::VectorXd &velocitiesAtStartOfTimestep,
    const Eigen::VectorXd &vertexMasses,
    double timestepSize,
    bool useGravity = true) {

  double h = timestepSize;
  Eigen::VectorXd x0 = positionsAtStartOfTimestep;
  Eigen::VectorXd v0 = velocitiesAtStartOfTimestep;
  int vertexCount = x0.size() / 3;

  // Create the kinetic potential
  Eigen::VectorXd predictivePositionsFlat = x0 + h * v0;

  if (useGravity) {
    double standard_gravity = -9.81; /* in m/s^2 */
    Eigen::MatrixXd gravityAccelerations = Eigen::MatrixXd::Zero(vertexCount, 3);
    gravityAccelerations.col(2) = standard_gravity * Eigen::VectorXd::Ones(vertexCount);
    Eigen::VectorXd a0 = silk::flatten(gravityAccelerations);

    predictivePositionsFlat += h * h * a0;
  }

  Eigen::MatrixXd predictivePositions = silk::unflatten(predictivePositionsFlat);

  // TODO: find fix for the formatting
  auto scalarFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));
  scalarFunction.add_elements<1>(TinyAD::range(vertexCount),
                                 [&, predictivePositions](auto &element) -> TINYAD_SCALAR_TYPE(element) {
                                   using ScalarT = TINYAD_SCALAR_TYPE(element);
                                   int vertexIndex = element.handle;

                                   Eigen::Vector3<ScalarT> position = element.variables(vertexIndex);
                                   Eigen::Vector3d predictivePosition = predictivePositions.row(vertexIndex);
                                   Eigen::Vector3<ScalarT> difference = position - predictivePosition;

                                   double vertexMass = vertexMasses(vertexIndex);

                                   ScalarT potential = 0.5 * vertexMass * difference.transpose() * difference;

                                   return potential;
                                 });

  return scalarFunction;
}

TinyAD::ScalarFunction<3, double, Eigen::Index> createVertexEnergyFunction(
    int vertexCount,
    const Points &scriptedVerticesIndices,
    const map<int, Eigen::Vector3d> &scriptedPositions,
    const Eigen::VectorXd &vertexMasses) {

  auto scalarFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));
  scalarFunction.add_elements<1>(
      TinyAD::range(scriptedVerticesIndices.size()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
        using ScalarT = TINYAD_SCALAR_TYPE(element);
        // TODO: possibly prevent the line below by using a std::vector instead of TinyAD::range
        int vertexIndex = scriptedVerticesIndices(element.handle);
        std::cout << vertexIndex << std::endl;
        Eigen::Vector3<ScalarT> position = element.variables(vertexIndex);
        Eigen::Vector3d scriptedPosition = scriptedPositions.at(vertexIndex);
        Eigen::Vector3<ScalarT> difference = position - scriptedPosition;

        double vertexMass = vertexMasses(vertexIndex);

        ScalarT potential = 0.5 * 1000000.0 * vertexMass * difference.transpose() * difference;
        return potential;
      });

  return scalarFunction;
}

template<typename FunctionTemplate>
EnergyFunction createTriangleEnergyFunction(FunctionTemplate &&triangleEnergy,
                                            int vertexCount,
                                            const Triangles &triangles,
                                            const vector<Eigen::Matrix2d> &triangleInvertedRestShapes) {

  auto scalarFunction = createTriangleScalarFunction(
      triangleEnergy, vertexCount, triangles, triangleInvertedRestShapes);
  EnergyFunction energyFunction = standardizeScalarFunction(scalarFunction);
  return energyFunction;
}

}  // namespace silk