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

#include "silk/conversions.hh"

#include "silk/energies.hh"
#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

using namespace silk;

geometrycentral::Vector3 to_geometrycentral(const Eigen::Vector3d &_v) {
  return geometrycentral::Vector3{_v.x(), _v.y(), _v.z()};
}

void drawSimulatedMesh(int frame,
                       ManifoldSurfaceMesh &mesh,
                       VertexPositionGeometry &geometry,
                       vector<VertexData<Eigen::Vector3d>> &positionsHistory,
                       map<string, vector<VertexData<Eigen::Vector3d>>> &vector3dVertexDataHistories) {

  for (Vertex v : mesh.vertices()) {
    geometry.vertexPositions[v] = to_geometrycentral(positionsHistory[frame][v]);
  }

  // Ensures refresh
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());

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
  std::tie(mesh_pointer, geometry_pointer) = silk::makeSquare();
  ManifoldSurfaceMesh &mesh = *mesh_pointer;
  VertexPositionGeometry &geometry = *geometry_pointer;

  geometry.requireVertexPositions();
  VertexData<Eigen::Vector2d> vertexRestPositions = silk::makeProjectedRestPositions(mesh, geometry);

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
    T stretchPotential = silk::baraffWitkinStretchPotential(F);
    T shearPotential = silk::baraffWitkinShearPotential(F);

    Eigen::Vector3d restNormal = silk::make3D(u1 - u0).cross(silk::make3D(u2 - u0));
    double restArea = restNormal.norm() / 2.0;

    double stretchStiffness = 100.0;
    double shearStiffness = 1.0;

    double areaFactor = sqrt(restArea);
    T shearEnergy = shearStiffness * areaFactor * shearPotential;
    T stretchEnergy = stretchStiffness * areaFactor * stretchPotential;
    T energy = stretchEnergy + shearEnergy;

    return energy;
  });

  Eigen::VectorXd initialPositions = meshTriangleEnergies.x_from_data(
      [&](Vertex v) { return to_eigen(geometry.vertexPositions[v]); });

  int vertexCount = mesh.nVertices();
  int system_size = initialPositions.size();
  Eigen::VectorXd initialVelocities = Eigen::VectorXd::Zero(system_size);
  // initialVelocities[6] = 0.1;

  vector<Eigen::VectorXd> positionsHistory;
  vector<Eigen::VectorXd> velocitiesHistory;
  vector<Eigen::VectorXd> forcesHistory;

  positionsHistory.push_back(initialPositions);
  velocitiesHistory.push_back(initialVelocities);

  double standard_gravity = -9.81; /* in m/s^2 */
  double dt = 0.005;

  // For the simulation part, we mostly work with flat (3*n_vertices, 1) column vectors.
  Eigen::VectorXd positions = initialPositions;
  Eigen::VectorXd velocities = initialVelocities;
  // Eigen::VectorXd forces = Eigen::VectorXd::Ones(system_size);
  double vertex_mass = 1.0 / system_size;
  Eigen::VectorXd masses = vertex_mass * Eigen::VectorXd::Ones(system_size);

  Eigen::SparseMatrix<double> identity_matrix(system_size, system_size);
  identity_matrix.setIdentity();

  Eigen::SparseMatrix<double> mass_matrix(system_size, system_size);
  mass_matrix.setIdentity();
  mass_matrix *= vertex_mass;

  // Set up gravity
  Eigen::VectorXd gravityAccelerations = Eigen::VectorXd::Zero(system_size);
  gravityAccelerations(Eigen::seqN(2, vertexCount, 3)) = standard_gravity * Eigen::VectorXd::Ones(vertexCount);
  Eigen::VectorXd gravityForces = mass_matrix * gravityAccelerations;

  std::vector<int> pinnedVertices;
  pinnedVertices.push_back(1);

  // TODO document
  Eigen::SparseMatrix<double> S(system_size, system_size);
  S.setIdentity();
  S.coeffRef(0, 0) = 0;
  S.coeffRef(1, 1) = 0;
  S.coeffRef(2, 2) = 0;
  S.coeffRef(3, 3) = 0;
  S.coeffRef(4, 4) = 0;
  S.coeffRef(5, 5) = 0;
  Eigen::VectorXd z = Eigen::VectorXd::Zero(system_size);

  // for (int i : pinnedVertices) {
  //   S.block(i, i, 3, 3) = 0.0;
  // }

  for (int i = 0; i < 200; i++) {

    auto [energy, gradient, projectedHessian] = meshTriangleEnergies.eval_with_hessian_proj(positions);
    std::cout << "Energy: " << energy << std::endl;
    Eigen::VectorXd forces = -gradient;

    double h = dt;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;
    Eigen::VectorXd f0 = forces;
    Eigen::SparseMatrix<double> dfdx = -projectedHessian;
    Eigen::SparseMatrix<double> M = mass_matrix;

    f0 += gravityForces;

    Eigen::SparseMatrix<double> A = M - (h * h) * dfdx;
    Eigen::VectorXd b = h * (f0 + h * (dfdx * v0));

    Eigen::SparseMatrix<double> I = identity_matrix;

    // SparseMatrix<float> S = create_S_matrix();
    Eigen::SparseMatrix<double> ST = Eigen::SparseMatrix<double>(S.transpose());
    Eigen::SparseMatrix<double> LHS = (S * A * ST) + I - S;
    Eigen::VectorXd c = b - A * z;
    Eigen::VectorXd rhs = S * c;

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
    solver.setTolerance(0.01);

    Eigen::VectorXd y(system_size);

    solver.compute(LHS);
    y = solver.solve(rhs);

    Eigen::VectorXd delta_v = y + z;

    // Non-filtered way:
    // Eigen::VectorXd delta_v(system_size);
    // cg.compute(A);
    // delta_v = cg.solve(b);

    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error() << std::endl;
    // The explicit way:
    // Eigen::VectorXd accelerations = forces.array() / masses.array();
    // velocities += accelerations * dt;

    velocities += delta_v;
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
}