#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "silk/conversions.hh"
#include "silk/deformation/rest_shapes.hh"
#include "silk/energies.hh"
#include "silk/energy/energy_function.hh"
#include "silk/energy/energy_mapping.hh"
#include "silk/geometry/area.hh"
#include "silk/mesh_construction.hh"
#include "silk/optimization/line_search.hh"
#include "silk/simple_meshes.hh"
#include "silk/types.hh"
#include "silk/visualization.hh"

#include <igl/triangle/triangulate.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

int main() {
  VertexPositions vertexPositions;
  vector<Points> pointGroups;
  vector<Edges> edgeGroups;
  vector<Triangles> triangleGroups;
  vector<Tetrahedra> tetrahedraGroups;

  // Add cloth
  auto [clothVertices, clothTriangles] = silk::makeRegularTriangle();

  Eigen::Matrix3d rotationMatrix = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()).matrix();
  clothVertices = (rotationMatrix * clothVertices.transpose()).transpose();
  clothTriangles = silk::appendTriangles(vertexPositions, triangleGroups, clothVertices, clothTriangles);

  vector<Eigen::Matrix2d> triangleRestShapes = silk::makeRestShapesFromCurrentPositions(vertexPositions,
                                                                                        clothTriangles);

  vector<Eigen::Matrix2d> triangleInvertedRestShapes;
  for (auto &restShape : triangleRestShapes) {
    triangleInvertedRestShapes.push_back(restShape.inverse());
  }

  map<string, silk::Energy *> energies;

  int vertexCount = vertexPositions.rows();

  Eigen::VectorXd triangleRestAreas = silk::triangleAreas(vertexPositions, clothTriangles);
  Eigen::VectorXd triangleStretchStiffnesses = Eigen::VectorXd::Ones(clothTriangles.rows());

  // Adding the stretch energy
  TinyAD::ScalarFunction<3, double, Eigen::Index> triangleStretchScalarFunction = silk::createTriangleScalarFunction(
      [](auto &&F) { return 100.0 * silk::baraffWitkinStretchPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);

  silk::TinyADEnergy stretchEnergy(triangleStretchScalarFunction);
  energies["triangleStretch"] = &stretchEnergy;

  // Adding the shear energy
  TinyAD::ScalarFunction<3, double, Eigen::Index> triangleShearScalarFunction = silk::createTriangleScalarFunction(
      [](auto &&F) { return 10.0 * silk::baraffWitkinShearPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);
  silk::TinyADEnergy shearEnergy(triangleShearScalarFunction);
  energies["triangleShear"] = &shearEnergy;

  int timesteps = 200;
  double timestepSize = 0.01;

  vertexPositions(0, 0) += 0.1;  // Small deformation

  vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory;
  vertexPositionHistory.push_back(vertexPositions);

  Eigen::VectorXd positions = silk::flatten(vertexPositions);
  Eigen::VectorXd velocities = Eigen::VectorXd::Zero(positions.size());

  vector<map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>> vertexVectorQuantitiesHistory;

  double vertexMass = 1.0 / vertexCount;
  Eigen::VectorXd vertexMasses = vertexMass * Eigen::VectorXd::Ones(vertexCount);

  for (int timestep = 0; timestep < timesteps; timestep++) {
    map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;

    double h = timestepSize;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;
    // Eigen::VectorXd a0 = gravityAccelerationsFlat;

    // Create the kinetic potential
    Eigen::VectorXd predictivePositionsFlat = x0 + h * v0;  // + h * h * a0;
    Eigen::MatrixXd predictivePositions = silk::unflatten(predictivePositionsFlat);
    auto kineticPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));
    kineticPotentialFunction.add_elements<1>(
        TinyAD::range(vertexCount), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
          using ScalarT = TINYAD_SCALAR_TYPE(element);
          int vertexIndex = element.handle;
          Eigen::Vector3<ScalarT> position = element.variables(vertexIndex);
          Eigen::Vector3d predictivePosition = predictivePositions.row(vertexIndex);
          Eigen::Vector3<ScalarT> difference = position - predictivePosition;

          double vertexMass = vertexMasses(vertexIndex);

          ScalarT potential = 0.5 * vertexMass * difference.transpose() * difference;

          return potential;
        });
    silk::TinyADEnergy kineticPotential(kineticPotentialFunction);
    energies["kineticPotential"] = &kineticPotential;

    for (auto const &[name, energy_ptr] : energies) {
      silk::Energy &energy = *energy_ptr;
      auto [f, g, H] = energy.eval_with_hessian_proj(x0);
      cout << name << ": " << f << endl;
      vertexVectorQuantities[name] = silk::unflatten(-g);
    }

    auto x = x0;
    int maxNewtonIterations = 50;
    double convergenceAccuracy = 1e-5;

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> conjugateGradientSolver;
    for (int newtonIteration = 0; newtonIteration < maxNewtonIterations; ++newtonIteration) {
      std::cout << "Newton iteration: " << newtonIteration << std::endl;

      auto [stretchValue, stretchGradient, stretchHessian] = stretchEnergy.eval_with_hessian_proj(x);
      auto [shearValue, shearGradient, shearHessian] = shearEnergy.eval_with_hessian_proj(x);
      auto [kineticValue, kineticGradient, kineticHessian] = kineticPotential.eval_with_hessian_proj(x);

      auto elasticValue = stretchValue + shearValue;
      auto elasticGradient = stretchGradient + shearGradient;
      auto elasticHessian = stretchHessian + shearHessian;

      auto incrementalPotential = kineticValue + h * h * elasticValue;
      auto incrementalPotentialGradient = kineticGradient + h * h * elasticGradient;
      auto incrementalPotentialHessian = kineticHessian + h * h * elasticHessian;

      double f = incrementalPotential;
      Eigen::VectorXd g = incrementalPotentialGradient;
      auto H_proj = incrementalPotentialHessian;
      std::function<double(const Eigen::VectorXd &)> func = [&](const Eigen::VectorXd &x) {
        return kineticPotential(x) + h * h * (stretchEnergy(x) + shearEnergy(x));
      };

      // Projected Netwon's method (line search with projected Hessian)
      Eigen::VectorXd d = conjugateGradientSolver.compute(H_proj).solve(-g);

      if (silk::convergenceConditionIPC(d, h)) {
        break;
      }
      x = silk::lineSearch(x, d, f, g, func);
    }

    positions = x;
    velocities = (x - x0).array() / h;
    vertexPositionHistory.push_back(silk::unflatten(positions));
    vertexVectorQuantitiesHistory.push_back(vertexVectorQuantities);
  }

  // Add empty map for the last timestep.
  vertexVectorQuantitiesHistory.push_back(map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>());

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 5.;
  // polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{{-2., -2., -2.}, {2., 2., 2.}};

  polyscope::state::userCallback = [&]() -> void {
    silk::playHistoryCallback(vertexPositionHistory,
                              pointGroups,
                              edgeGroups,
                              triangleGroups,
                              tetrahedraGroups,
                              vertexVectorQuantitiesHistory);
  };

  polyscope::show();
}