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
using namespace std::chrono;

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
      [](auto &&F) { return 20.0 * silk::baraffWitkinStretchPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);

  silk::TinyADEnergy stretchEnergy(triangleStretchScalarFunction);
  energies["triangleStretch"] = &stretchEnergy;

  // Adding the shear energy
  TinyAD::ScalarFunction<3, double, Eigen::Index> triangleShearScalarFunction = silk::createTriangleScalarFunction(
      [](auto &&F) { return 2.0 * silk::baraffWitkinShearPotential(F); },
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

  // Setting up the energy for the scripted vertices
  Points staticVertices(1);
  staticVertices << 2;
  map<int, Eigen::Vector3d> scriptedPositions;
  for (int index : staticVertices) {
    scriptedPositions[index] = vertexPositions.row(index);
  }

  TinyAD::ScalarFunction<3, double, Eigen::Index> staticSpringEnergy = silk::createVertexEnergyFunction(
      vertexCount, staticVertices, scriptedPositions, vertexMasses);
  silk::TinyADEnergy staticSpring(staticSpringEnergy);
  energies["staticSpring"] = &staticSpring;

  auto start = high_resolution_clock::now();

  for (int timestep = 0; timestep < timesteps; timestep++) {
    map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;

    double h = timestepSize;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;

    TinyAD::ScalarFunction<3, double, Eigen::Index> kineticPotentialFunction = silk::createKineticPotentialFunction(
        x0, v0, vertexMasses, h);
    silk::TinyADEnergy kineticPotential(kineticPotentialFunction);
    energies["kineticPotential"] = &kineticPotential;

    for (auto const &[name, energy_ptr] : energies) {
      silk::Energy &energy = *energy_ptr;
      auto [f, g, H] = energy.eval_with_hessian_proj(x0);
      cout << name << ": " << f << endl;
      vertexVectorQuantities[name] = silk::unflatten(-g);
    }

    // TODO: think about how to ensure/enforce all energies have a weight.
    map<string, double> energyWeights = {
        {"kineticPotential", 1.0}, {"triangleStretch", h * h}, {"triangleShear", h * h}, {"staticSpring", h * h}};
    silk::AdditiveEnergy incrementalPotential(energies, energyWeights);

    auto x = x0;
    int maxNewtonIterations = 50;
    double convergenceAccuracy = 1e-5;

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> conjugateGradientSolver;
    for (int newtonIteration = 0; newtonIteration < maxNewtonIterations; ++newtonIteration) {
      std::cout << "Newton iteration: " << newtonIteration << std::endl;

      auto [f, g, H_proj] = incrementalPotential.eval_with_hessian_proj(x);
      std::function<double(const Eigen::VectorXd &)> func = [&](const Eigen::VectorXd &x) {
        return incrementalPotential(x);
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

  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<seconds>(stop - start);
  cout << "Time taken: " << duration.count() << " seconds " << endl;

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