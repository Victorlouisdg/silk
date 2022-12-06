#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

#include "silk/energies.hh"
#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <igl/boundary_facets.h>
#include <igl/edges.h>
#include <igl/triangle/triangulate.h>

#include <ipc/ipc.hpp>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

geometrycentral::Vector3 to_geometrycentral(const Eigen::Vector3d &_v) {
  return geometrycentral::Vector3{_v.x(), _v.y(), _v.z()};
}

void drawSimulatedTetMesh(int frame,
                          Eigen::MatrixXi T,
                          vector<Eigen::MatrixXd> &positionsHistory,
                          map<string, vector<Eigen::MatrixXd>> &vector3dVertexDataHistories) {

  // for (Vertex v : mesh.vertices()) {
  //   geometry.vertexPositions[v] = to_geometrycentral(positionsHistory[frame][v]);
  // }

  Eigen::MatrixXd V = positionsHistory[frame];

  // Ensures refresh

  auto *psMesh = polyscope::registerTetMesh("my mesh", V, T);

  // auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());

  for (auto const &[name, dataHistory] : vector3dVertexDataHistories) {
    if (frame >= dataHistory.size()) {
      continue;
    }
    psMesh->addVertexVectorQuantity(name, dataHistory[frame]);
  }
}

// std::vector<Eigen::Index>

void callback(Eigen::MatrixXi T,
              vector<Eigen::MatrixXd> &positionsHistory,
              map<string, vector<Eigen::MatrixXd>> &vector3dVertexDataHistories) {

  int frame = silk::state::playback_frame_counter % positionsHistory.size();

  ImGui::Text("Frame %d", frame);

  if (ImGui ::Button("Next frame")) {
    silk::state::playback_frame_counter++;
    frame = silk::state::playback_frame_counter % positionsHistory.size();
    drawSimulatedTetMesh(frame, T, positionsHistory, vector3dVertexDataHistories);
  }

  if (ImGui ::Button("Previous frame")) {
    silk::state::playback_frame_counter--;
    frame = silk::state::playback_frame_counter % positionsHistory.size();
    drawSimulatedTetMesh(frame, T, positionsHistory, vector3dVertexDataHistories);
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

  drawSimulatedTetMesh(frame, T, positionsHistory, vector3dVertexDataHistories);
  silk::state::playback_frame_counter++;
}

vector<Eigen::MatrixXd> flatHistoryToIGL(vector<Eigen::VectorXd> flatVectorHistory,
                                         TinyAD::ScalarFunction<3, double, Eigen::Index> &tinyadFunction) {

  vector<Eigen::MatrixXd> vertexDataHistory;
  for (Eigen::VectorXd flatVector : flatVectorHistory) {
    Eigen::MatrixXd V(flatVector.rows() / 3, 3);
    tinyadFunction.x_to_data(flatVector,
                             [&](int v_idx, const Eigen::Vector3d &vectorData) { V.row(v_idx) = vectorData; });
    vertexDataHistory.push_back(V);
  }
  return vertexDataHistory;
}

//
int main() {
  Eigen::MatrixXd V;
  Eigen::MatrixXi T;
  std::tie(V, T) = silk::makeStackedTetrahedra();

  Eigen::MatrixXi F;
  igl::boundary_facets(T, F);

  Eigen::MatrixXi E;
  igl::edges(F, E);

  std::vector<Eigen::Matrix3d> restShapes = silk::initializeTetrahedronRestShapes(V, T);

  // Small squash
  // V(0, 2) -= 0.1;

  // Set up a function with 3D vertex positions as variables
  auto tetrahedronEnergies = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

  tetrahedronEnergies.add_elements<4>(TinyAD::range(T.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    // Evaluate element using either double or TinyAD::Double
    using ScalarT = TINYAD_SCALAR_TYPE(element);
    int t_idx = element.handle;
    Eigen::Vector3<ScalarT> a = element.variables(T(t_idx, 0));
    Eigen::Vector3<ScalarT> b = element.variables(T(t_idx, 1));
    Eigen::Vector3<ScalarT> c = element.variables(T(t_idx, 2));
    Eigen::Vector3<ScalarT> d = element.variables(T(t_idx, 3));
    Eigen::Matrix3<ScalarT> M = TinyAD::col_mat(b - a, c - a, d - a);

    // Compute tet Jacobian and energy
    Eigen::Matrix3d Mr = restShapes[t_idx];
    Eigen::Matrix3<ScalarT> J = M * Mr.inverse();
    double vol = Mr.determinant() / 6.0;

    // The exponential symmetric Dirichlet energy:
    return vol * exp((J.squaredNorm() + J.inverse().squaredNorm()));
  });

  Eigen::VectorXd initialPositions = tetrahedronEnergies.x_from_data([&](int v_idx) { return V.row(v_idx); });

  int vertexCount = V.rows();
  int system_size = initialPositions.size();
  Eigen::VectorXd initialVelocities = Eigen::VectorXd::Zero(system_size);
  initialVelocities(4 * 3 + 2) = -2.0;  // v_z of top vertex of upper tetrahedron

  vector<Eigen::VectorXd> positionsHistory;
  vector<Eigen::VectorXd> velocitiesHistory;
  vector<Eigen::VectorXd> forcesHistory;

  positionsHistory.push_back(initialPositions);
  velocitiesHistory.push_back(initialVelocities);

  double dt = 0.005;

  Eigen::VectorXd positions = initialPositions;
  Eigen::VectorXd velocities = initialVelocities;

  double vertex_mass = 1.0 / system_size;
  Eigen::VectorXd masses = vertex_mass * Eigen::VectorXd::Ones(system_size);

  Eigen::SparseMatrix<double> identity_matrix(system_size, system_size);
  identity_matrix.setIdentity();

  Eigen::SparseMatrix<double> mass_matrix(system_size, system_size);
  mass_matrix.setIdentity();
  mass_matrix *= vertex_mass;

  for (int i = 0; i < 200; i++) {

    auto [energy, gradient, projectedHessian] = tetrahedronEnergies.eval_with_hessian_proj(positions);
    std::cout << "Energy: " << energy << std::endl;
    Eigen::VectorXd forces = -gradient;

    // update V
    tetrahedronEnergies.x_to_data(positions, [&](int v_idx, const Eigen::Vector3d &v) { V.row(v_idx) = v; });
    ipc::CollisionMesh mesh(V, E, F);

    Eigen::MatrixXd collisionV = mesh.vertices(V);
    ipc::BroadPhaseMethod method = ipc::BroadPhaseMethod::BRUTE_FORCE;
    ipc::Constraints constraint_set;
    double dhat = 0.2;  // square of maximum distance at which repulsion works
    constraint_set.build(mesh, collisionV, dhat, /*dmin=*/0, method);

    Eigen::VectorXd grad_b = ipc::compute_barrier_potential_gradient(mesh, collisionV, constraint_set, dhat);
    Eigen::SparseMatrix<double> hess_b = ipc::compute_barrier_potential_hessian(
        mesh, V, constraint_set, dhat, /*project_to_psd=*/true);

    double h = dt;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;
    Eigen::VectorXd f0 = forces - grad_b;
    Eigen::SparseMatrix<double> dfdx = -projectedHessian - hess_b;
    Eigen::SparseMatrix<double> M = mass_matrix;

    Eigen::SparseMatrix<double> A = M - (h * h) * dfdx;
    Eigen::VectorXd b = h * (f0 + h * (dfdx * v0));

    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Lower | Eigen::Upper> solver;
    solver.setTolerance(0.01);

    Eigen::VectorXd delta_v(system_size);

    solver.compute(A);
    delta_v = solver.solve(b);

    std::cout << "#iterations:     " << solver.iterations() << std::endl;
    std::cout << "estimated error: " << solver.error() << std::endl;

    velocities += delta_v;
    positions += velocities * dt;

    positionsHistory.push_back(positions);
    velocitiesHistory.push_back(velocities);
    forcesHistory.push_back(forces);
  }

  map<string, vector<Eigen::MatrixXd>> vector3dVertexDataHistories;
  vector3dVertexDataHistories["velocities"] = flatHistoryToIGL(velocitiesHistory, tetrahedronEnergies);
  vector3dVertexDataHistories["forces"] = flatHistoryToIGL(forcesHistory, tetrahedronEnergies);

  vector<Eigen::MatrixXd> positionsHistoryData = flatHistoryToIGL(positionsHistory, tetrahedronEnergies);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 1.;
  polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{{-5., -5., -5.}, {5., 5., 5.}};

  polyscope::registerTetMesh("my mesh", V, T);
  polyscope::state::userCallback = [&]() -> void { callback(T, positionsHistoryData, vector3dVertexDataHistories); };
  polyscope::show();
}