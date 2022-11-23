#include <TinyAD/Support/GeometryCentral.hh>

#include <TinyAD/Scalar.hh>
#include <TinyAD/ScalarFunction.hh>

#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include <TinyAD-Examples/IGLGeometryCentralConvert.hh>

#include "silk/energies.hh"
#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

geometrycentral::Vector3 to_geometrycentral(const Eigen::Vector3d &_v) {
  return geometrycentral::Vector3{_v.x(), _v.y(), _v.z()};
}

VertexData<Eigen::Vector2d> makeRestPositionsProjected(ManifoldSurfaceMesh &mesh, VertexPositionGeometry &geometry) {
  geometry.requireVertexPositions();
  VertexData<Eigen::Vector2d> vertexRestPositions(mesh);
  for (Vertex v : mesh.vertices()) {
    Vector3 position = geometry.vertexPositions[v];
    vertexRestPositions[v] = Eigen::Vector2d({position.x, position.y});
  }
  return vertexRestPositions;
}

void callback(ManifoldSurfaceMesh &mesh,
              VertexPositionGeometry &geometry,
              vector<VertexData<Eigen::Vector3d>> &positionsHistory) {

  // Ensures refresh
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());
  int i = silk::state::gui_frame % positionsHistory.size();

  for (Vertex v : mesh.vertices()) {
    geometry.vertexPositions[v] = to_geometrycentral(positionsHistory[i][v]);
  }
  silk::state::gui_frame++;
}

int main() {
  unique_ptr<ManifoldSurfaceMesh> mesh_pointer;
  unique_ptr<VertexPositionGeometry> geometry_pointer;
  std::tie(mesh_pointer, geometry_pointer) = silk::makeTwoTriangleSquare();
  ManifoldSurfaceMesh &mesh = *mesh_pointer;
  VertexPositionGeometry &geometry = *geometry_pointer;

  geometry.requireVertexPositions();
  VertexData<Eigen::Vector2d> vertexRestPositions = makeRestPositionsProjected(mesh, geometry);

  // Set up a function with 3D vertex positions as variables
  auto meshTriangleEnergies = TinyAD::scalar_function<3>(mesh.vertices());

  // Add objective term per face. Each connecting 3 vertices.
  meshTriangleEnergies.add_elements<3>(mesh.faces(), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    // Evaluate element using either double or TinyAD::Double
    using T = TINYAD_SCALAR_TYPE(element);

    // Get variable 2D vertex positions
    Face f = element.handle;
    Vertex v0 = f.halfedge().vertex();
    Vertex v1 = f.halfedge().next().vertex();
    Vertex v2 = f.halfedge().next().next().vertex();
    Eigen::Vector3<T> x0 = element.variables(v0);
    Eigen::Vector3<T> x1 = element.variables(v1);
    Eigen::Vector3<T> x2 = element.variables(v2);

    Eigen::Vector2d u0 = vertexRestPositions[v0];
    Eigen::Vector2d u1 = vertexRestPositions[v1];
    Eigen::Vector2d u2 = vertexRestPositions[v2];

    Eigen::Matrix<T, 3, 2> F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
    T E = silk::baraffWitkinStretchEnergy(F);
    return E;
  });

  Eigen::VectorXd initialPositions = meshTriangleEnergies.x_from_data(
      [&](Vertex v) { return to_eigen(geometry.vertexPositions[v]); });

  int system_size = initialPositions.size();
  Eigen::VectorXd initialVelocities = Eigen::VectorXd::Zero(system_size);
  initialVelocities[2] = 0.01;

  vector<Eigen::VectorXd> positionsHistory;
  vector<Eigen::VectorXd> velocitiesHistory;
  vector<Eigen::VectorXd> energyHistory;
  positionsHistory.push_back(initialPositions);

  double dt = 0.1;

  Eigen::VectorXd positions = initialPositions;
  Eigen::VectorXd velocities = initialVelocities;
  for (int i = 0; i < 50; i++) {
    positions += velocities * dt;
    positionsHistory.push_back(positions);
  }

  // map<string, VertexData<double>> vertexData;
  // vertexData["velocities"]

  vector<VertexData<Eigen::Vector3d>> positionsHistoryData;

  for (Eigen::VectorXd positionsFlat : positionsHistory) {
    VertexData<Eigen::Vector3d> positionsData(mesh);
    meshTriangleEnergies.x_to_data(positionsFlat, [&](Vertex v, const Eigen::Vector3d &p) { positionsData[v] = p; });
    positionsHistoryData.push_back(positionsData);
  }

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());
  // psMesh->addVertexParameterizationQuantity("restPosition", vertexRestPositions);
  polyscope::state::userCallback = [&]() -> void { callback(mesh, geometry, positionsHistoryData); };
  polyscope::show();

  // auto [E, g, H_proj] = meshEnergyFunction.eval_with_hessian_proj(x);

  // Eigen::VectorXd f0 = g;
  // Eigen::SparseMatrix<double> dfdx = H_proj;

  // int amount_of_vertices = geometry->vertexPositions.size();
  // int system_size = 3 * amount_of_vertices;

  // Eigen::VectorXd masses = Eigen::VectorXd::Ones(system_size);

  // std::cout << masses << std::endl;

  // Eigen::VectorXd accelerations = f0.array() / masses.array();
  // Eigen::VectorXd velocities = Eigen::VectorXd::Zero(system_size);
  // double dt = 0.01;
  // velocities += accelerations * dt;

  // Eigen::VectorXd positions = x;
  // positions += velocities * dt;

  // std::cout << "x:\n" << x << std::endl;
  // std::cout << "positions:\n" << positions << std::endl;

  // Eigen::SparseMatrix<double> identity_matrix(system_size, system_size);
  // identity_matrix.setIdentity();
}