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
      [](auto &&F) { return silk::baraffWitkinStretchPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);

  silk::TinyADEnergy stretchEnergy(triangleStretchScalarFunction);
  energies["triangleStretch"] = &stretchEnergy;

  vertexPositions(0, 0) += 0.1;
  Eigen::VectorXd x = silk::flatten(vertexPositions);

  map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;
  for (auto const &[name, energy_ptr] : energies) {
    silk::Energy &energy = *energy_ptr;
    auto [f, g, H] = energy.eval_with_hessian_proj(x);
    cout << name << ": " << f << endl;
    vertexVectorQuantities[name] = silk::unflatten(-g);
  }

  // int timesteps = 20;
  // double timestepSize = 0.01;

  // vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory;
  // vertexPositionHistory.push_back(vertexPositions);

  // Eigen::VectorXd positions = silk::flatten(vertexPositions);
  // Eigen::VectorXd velocities = Eigen::VectorXd::Zero(positions.size());

  // vector<map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>> vertexVectorQuantitiesHistory;

  // double vertexMass = 1.0 / vertexCount;
  // Eigen::VectorXd vertexMasses = vertexMass * Eigen::VectorXd::Ones(vertexCount);

  // for (int timestep = 0; timestep < timesteps; timestep++) {
  //   map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;

  //   double h = timestepSize;
  //   Eigen::VectorXd x0 = positions;
  //   Eigen::VectorXd v0 = velocities;
  //   // Eigen::VectorXd a0 = gravityAccelerationsFlat;

  //   Eigen::VectorXd predictivePositionsFlat = x0 + h * v0;  // + h * h * a0;
  //   Eigen::MatrixXd predictivePositions = silk::unflatten(predictivePositionsFlat);

  //   auto x = x0;
  //   int maxNewtonIterations = 50;
  //   double convergenceEps = 1e-5;

  //   Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> conjugateGradientSolver;
  //   for (int newtonIteration = 0; newtonIteration < maxNewtonIterations; ++newtonIteration) {
  //   }
  // }

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 5.;
  silk::registerInPolyscope(
      vertexPositions, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups, vertexVectorQuantities);
  polyscope::show();
}