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

void drawSimulatedMesh(int frame,
                       ManifoldSurfaceMesh &mesh,
                       VertexPositionGeometry &geometry,
                       vector<VertexData<Eigen::Vector3d>> &positionsHistory,
                       map<string, vector<VertexData<Eigen::Vector3d>>> &vector3dVertexDataHistories) {
  // Ensures refresh
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());

  for (Vertex v : mesh.vertices()) {
    geometry.vertexPositions[v] = to_geometrycentral(positionsHistory[frame][v]);
  }

  for (auto const &[name, dataHistory] : vector3dVertexDataHistories) {
    if (frame >= dataHistory.size()) {
      continue;
    }
    psMesh->addVertexVectorQuantity(name, dataHistory[frame]);
  }
}

void callback(ManifoldSurfaceMesh &mesh,
              VertexPositionGeometry &geometry,
              vector<VertexData<Eigen::Vector3d>> &positionsHistory,
              map<string, vector<VertexData<Eigen::Vector3d>>> &vector3dVertexDataHistories) {

  int frame = silk::state::playback_frame_counter % positionsHistory.size();

  ImGui::Text("Frame %d", frame);

  if (ImGui ::Button("Next frame")) {
    silk::state::playback_frame_counter++;
    frame = silk::state::playback_frame_counter % positionsHistory.size();
    drawSimulatedMesh(frame, mesh, geometry, positionsHistory, vector3dVertexDataHistories);
  }

  if (ImGui ::Button("Previous frame")) {
    silk::state::playback_frame_counter--;
    frame = silk::state::playback_frame_counter % positionsHistory.size();
    drawSimulatedMesh(frame, mesh, geometry, positionsHistory, vector3dVertexDataHistories);
  }

  // Play-pause logic
  if (silk::state::playback_paused) {
    if (ImGui ::Button("Resume playback")) {
      silk::state::playback_paused = false;
    }
  } else {
    if (ImGui::Button("Pause playback")) {
      silk::state::playback_paused = true;
    }
  }
  if (silk::state::playback_paused) {
    return;
  }

  drawSimulatedMesh(frame, mesh, geometry, positionsHistory, vector3dVertexDataHistories);
  silk::state::playback_frame_counter++;
}

vector<VertexData<Eigen::Vector3d>> flatHistoryToGeometryCentral(
    vector<Eigen::VectorXd> flatVectorHistory,
    ManifoldSurfaceMesh &mesh,
    TinyAD::ScalarFunction<3, double, Vertex> &tinyadFunction) {

  vector<VertexData<Eigen::Vector3d>> vertexDataHistory;
  for (Eigen::VectorXd flatVector : flatVectorHistory) {
    VertexData<Eigen::Vector3d> vertexData(mesh);
    tinyadFunction.x_to_data(flatVector,
                             [&](Vertex v, const Eigen::Vector3d &vectorData) { vertexData[v] = vectorData; });
    vertexDataHistory.push_back(vertexData);
  }
  return vertexDataHistory;
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
  auto meshTriangleEnergies = TinyAD::scalar_function<3, double, VertexSet>(mesh.vertices());

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
  initialVelocities[0] = 0.1;

  vector<Eigen::VectorXd> positionsHistory;
  vector<Eigen::VectorXd> velocitiesHistory;
  vector<Eigen::VectorXd> forcesHistory;

  positionsHistory.push_back(initialPositions);
  velocitiesHistory.push_back(initialVelocities);

  double dt = 0.1;

  // For the simulation part, we mostly work with flat (3*n_vertices, 1) column vectors.
  Eigen::VectorXd positions = initialPositions;
  Eigen::VectorXd velocities = initialVelocities;
  Eigen::VectorXd forces = 0.1 * Eigen::VectorXd::Ones(system_size);
  Eigen::VectorXd masses = Eigen::VectorXd::Ones(system_size);

  for (int i = 0; i < 100; i++) {

    auto [energy, gradient, projectedHessian] = meshTriangleEnergies.eval_with_hessian_proj(positions);
    std::cout << "Energy: " << energy << std::endl;
    Eigen::VectorXd forces = -gradient;

    Eigen::VectorXd accelerations = forces.array() / masses.array();
    velocities += accelerations * dt;
    positions += velocities * dt;

    positionsHistory.push_back(positions);
    velocitiesHistory.push_back(velocities);
    forcesHistory.push_back(forces);
  }

  map<string, vector<VertexData<Eigen::Vector3d>>> vector3dVertexDataHistories;
  vector3dVertexDataHistories["velocities"] = flatHistoryToGeometryCentral(
      velocitiesHistory, mesh, meshTriangleEnergies);
  vector3dVertexDataHistories["forces"] = flatHistoryToGeometryCentral(forcesHistory, mesh, meshTriangleEnergies);

  vector<VertexData<Eigen::Vector3d>> positionsHistoryData = flatHistoryToGeometryCentral(
      positionsHistory, mesh, meshTriangleEnergies);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 1.;
  polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{{-2., -2., -2.}, {2., 2., 2.}};
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());
  // psMesh->addVertexParameterizationQuantity("restPosition", vertexRestPositions);
  polyscope::state::userCallback = [&]() -> void {
    callback(mesh, geometry, positionsHistoryData, vector3dVertexDataHistories);
  };
  polyscope::show();

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