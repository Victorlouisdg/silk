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

#include <fsim/ElasticShell.h>

using namespace std;

int main() {
  VertexPositions vertexPositions;
  vector<Points> pointGroups;
  vector<Edges> edgeGroups;
  vector<Triangles> triangleGroups;
  vector<Tetrahedra> tetrahedraGroups;

  // Add cloth
  auto [clothVertices, clothTriangles] = silk::makeRegularTriangle();

  // Eigen::Matrix3d rotationMatrix = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()).matrix();
  // clothVertices = (rotationMatrix * clothVertices.transpose()).transpose();

  clothVertices.conservativeResize(clothVertices.rows() + 1, 3);
  clothVertices.row(clothVertices.rows() - 1) = clothVertices.row(2);
  clothVertices(clothVertices.rows() - 1, 1) *= -1.0;

  clothTriangles.conservativeResize(clothTriangles.rows() + 1, 3);
  clothTriangles.row(clothTriangles.rows() - 1) = Eigen::Vector3i(clothVertices.rows() - 1, 1, 0);

  clothTriangles = silk::appendTriangles(vertexPositions, triangleGroups, clothVertices, clothTriangles);

  fsim::DiscreteShell<fsim::TanAngleFormulation> shell(vertexPositions, clothTriangles, 1.0, 1.0, 0.5);

  // Bend the triangles upwards
  vertexPositions.bottomRows(2).col(2).array() += 0.5;

  const Eigen::VectorXd x = silk::flatten(vertexPositions);
  const Eigen::Ref<const Eigen::VectorXd> xRef(x);
  double energy = shell.energy(x);
  // Eigen::VectorXd gradient =

  Eigen::VectorXd y = Eigen::VectorXd::Zero(x.size());
  Eigen::Ref<Eigen::VectorXd> yRef(y);

  shell.gradient(xRef, yRef);

  std::cout << y << std::endl;

  std::cout << "Initial energy: " << energy << std::endl;

  map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;
  vertexVectorQuantities["forces"] = -silk::unflatten(y);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 5.;
  silk::registerInPolyscope(
      vertexPositions, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups, vertexVectorQuantities);

  polyscope::show();
}