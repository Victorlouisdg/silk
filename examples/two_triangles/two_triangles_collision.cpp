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

#include <igl/edges.h>

#include <ipc/ipc.hpp>

#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

using namespace silk;

void drawSimulatedMesh(int frame,
                       Eigen::MatrixXi &triangles,
                       vector<Eigen::MatrixXd> &positionsHistory,
                       map<string, vector<Eigen::MatrixXd>> &vector3dVertexDataHistories) {

  // Ensures refresh
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", positionsHistory[frame], triangles);

  for (auto const &[name, dataHistory] : vector3dVertexDataHistories) {
    if (frame >= dataHistory.size()) {
      continue;
    }
    psMesh->addVertexVectorQuantity(name, dataHistory[frame]);
  }
}

void callback(Eigen::MatrixXi &triangles,
              vector<Eigen::MatrixXd> &positionsHistory,
              map<string, vector<Eigen::MatrixXd>> &vector3dVertexDataHistories) {

  int frame = silk::state::playback_frame_counter % positionsHistory.size();

  ImGui::Text("Frame %d", frame);

  if (ImGui ::Button("Next frame")) {
    silk::state::playback_frame_counter++;
    frame = silk::state::playback_frame_counter % positionsHistory.size();
    drawSimulatedMesh(frame, triangles, positionsHistory, vector3dVertexDataHistories);
  }

  if (ImGui ::Button("Previous frame")) {
    silk::state::playback_frame_counter--;
    frame = silk::state::playback_frame_counter % positionsHistory.size();
    drawSimulatedMesh(frame, triangles, positionsHistory, vector3dVertexDataHistories);
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

  drawSimulatedMesh(frame, triangles, positionsHistory, vector3dVertexDataHistories);
  silk::state::playback_frame_counter++;
}

int main() {
  unique_ptr<SurfaceMesh> mesh_pointer;
  unique_ptr<VertexPositionGeometry> geometry_pointer;
  std::tie(mesh_pointer, geometry_pointer) = silk::makeOrthogonalTriangles();
  SurfaceMesh &mesh = *mesh_pointer;
  VertexPositionGeometry &geometry = *geometry_pointer;

  geometry.requireVertexPositions();
  // VertexData<Eigen::Vector2d> vertexRestPositions = silk::makeProjectedRestPositions(mesh, geometry);
  // MatrixXd vertexRestPositions;

  Eigen::MatrixXd vertexPositions;                     // 3D vertex positions
  Eigen::MatrixXi triangles;                           // Mesh faces
  to_igl(mesh, geometry, vertexPositions, triangles);  // Convert mesh to igl format

  Eigen::MatrixXi edges;
  igl::edges(triangles, edges);

  Eigen::MatrixXd vertexRestPositions = vertexPositions.leftCols<2>();
  vertexRestPositions.bottomRows(3) = vertexRestPositions.topRows(3);

  // Set up a function with 3D vertex positions as variables
  // auto meshTriangleEnergies = TinyAD::scalar_function<3, double, VertexSet>(mesh.vertices());

  auto elasticPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexPositions.rows()));

  // Add objective term per face. Each connecting 3 vertices.
  elasticPotentialFunction.add_elements<3>(
      TinyAD::range(triangles.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
        // Evaluate element using either double or TinyAD::Double
        using ScalarT = TINYAD_SCALAR_TYPE(element);
        int triangleIndex = element.handle;

        // Get vertex indices for this triangle
        Eigen::Vector3i triangle = triangles.row(triangleIndex);
        int v0 = triangle(0);
        int v1 = triangle(1);
        int v2 = triangle(2);

        Eigen::Vector3<ScalarT> x0 = element.variables(v0);
        Eigen::Vector3<ScalarT> x1 = element.variables(v1);
        Eigen::Vector3<ScalarT> x2 = element.variables(v2);

        Eigen::Vector2d u0 = vertexRestPositions.row(v0);
        Eigen::Vector2d u1 = vertexRestPositions.row(v1);
        Eigen::Vector2d u2 = vertexRestPositions.row(v2);

        Eigen::Matrix<ScalarT, 3, 2> F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
        ScalarT stretchPotential = silk::baraffWitkinStretchPotential(F);
        ScalarT shearPotential = silk::baraffWitkinShearPotential(F);

        Eigen::Vector3d restNormal = silk::make3D(u1 - u0).cross(silk::make3D(u2 - u0));
        double restArea = restNormal.norm() / 2.0;

        double stretchStiffness = 100.0;
        double shearStiffness = 1.0;

        double areaFactor = sqrt(restArea);
        ScalarT shearEnergy = shearStiffness * areaFactor * shearPotential;
        ScalarT stretchEnergy = stretchStiffness * areaFactor * stretchPotential;
        ScalarT energy = stretchEnergy + shearEnergy;

        return energy;
      });

  Eigen::VectorXd initialPositions = elasticPotentialFunction.x_from_data(
      [&](int index) { return vertexPositions.row(index); });

  // Eigen::VectorXd initialPositions = elasticPotentialFunction.x_from_data(
  //     [&](Vertex v) { return to_eigen(geometry.vertexPositions[v]); });

  int vertexCount = mesh.nVertices();
  int system_size = initialPositions.size();
  Eigen::VectorXd initialVelocities = Eigen::VectorXd::Zero(system_size);
  // initialVelocities[6] = 0.1;
  initialVelocities(3 * 3 + 2) = -4.0;

  // IGL-style Nx3 Eigen matrices to record the history of the simulation.
  vector<Eigen::MatrixXd> positionsHistory;
  vector<Eigen::MatrixXd> velocitiesHistory;
  vector<Eigen::MatrixXd> forcesHistory;

  // positionsHistory.push_back(initialPositions);
  // velocitiesHistory.push_back(initialVelocities);

  double standard_gravity = -9.81; /* in m/s^2 */
  double dt = 0.005;

  // For the simulation part, we mostly work with flat (3*n_vertices, 1) column vectors.
  Eigen::VectorXd positions = initialPositions;
  Eigen::VectorXd velocities = initialVelocities;

  double vertex_mass = 1.0 / system_size;
  Eigen::VectorXd masses = vertex_mass * Eigen::VectorXd::Ones(system_size);
  Eigen::SparseMatrix<double> identity_matrix(system_size, system_size);
  identity_matrix.setIdentity();
  Eigen::SparseMatrix<double> mass_matrix(system_size, system_size);
  mass_matrix.setIdentity();
  mass_matrix *= vertex_mass;
  Eigen::SparseMatrix<double> M = mass_matrix;
  Eigen::SparseMatrix<double> Minv = -M;

  ipc::CollisionMesh collisionMesh(vertexPositions, edges, triangles);
  ipc::BroadPhaseMethod method = ipc::BroadPhaseMethod::BRUTE_FORCE;
  ipc::Constraints constraintSet;
  double dhat = 0.01;  // square of maximum distance at which repulsion works

  for (int i = 0; i < 200; i++) {
    double h = dt;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;

    Eigen::VectorXd predictivePositionsFlat = x0 + h * v0;  //+ h * h;  // * Minv
    Eigen::MatrixXd predictivePositions(vertexCount, 3);
    elasticPotentialFunction.x_to_data(predictivePositionsFlat, [&](int v_idx, const Eigen::Vector3d &vectorData) {
      predictivePositions.row(v_idx) = vectorData;
    });

    auto kineticPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));
    kineticPotentialFunction.add_elements<1>(
        TinyAD::range(vertexCount), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
          // Evaluate element using either double or TinyAD::Double
          using ScalarT = TINYAD_SCALAR_TYPE(element);
          int v_idx = element.handle;
          Eigen::Vector3<ScalarT> position = element.variables(v_idx);
          Eigen::Vector3d predictivePosition = predictivePositions.row(v_idx);
          Eigen::Vector3<ScalarT> difference = position - predictivePosition;

          double vertex_mass = masses(v_idx);

          ScalarT potential = 0.5 * vertex_mass * difference.transpose() * difference;

          return potential;
        });

    auto kinetic = kineticPotentialFunction.eval(x0);
    auto [elastic, elasticGradient] = elasticPotentialFunction.eval_with_gradient(x0);
    std::cout << "k:" << kinetic << " e: " << elastic << std::endl;

    // Projected Newton with conjugate gradient solver
    auto x = x0;

    int max_iters = 50;
    double convergence_eps = 1e-8;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> cg_solver;
    for (int i = 0; i < max_iters; ++i) {
      // auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
      auto [elasticPotential,
            elasticPotentialGradient,
            elasticPotentialHessian] = elasticPotentialFunction.eval_with_hessian_proj(x);

      auto [kineticPotential,
            kineticPotentialGradient,
            kineticPotentialHessian] = kineticPotentialFunction.eval_with_hessian_proj(x);

      // Start for stuff to wrap
      // flat x in -> potential, (flat) gradient and hessian out.
      // unflatten vertex positions
      elasticPotentialFunction.x_to_data(x,
                                         [&](int v_idx, const Eigen::Vector3d &v) { vertexPositions.row(v_idx) = v; });
      Eigen::MatrixXd collisionV = collisionMesh.vertices(vertexPositions);
      constraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);

      // Evaluate barrier potential and derivatives
      double barrierPotential = ipc::compute_barrier_potential(collisionMesh, collisionV, constraintSet, dhat);
      Eigen::VectorXd barrierPotentialGradient = ipc::compute_barrier_potential_gradient(
          collisionMesh, collisionV, constraintSet, dhat);
      Eigen::SparseMatrix<double> barrierPotentialHessian = ipc::compute_barrier_potential_hessian(
          collisionMesh, collisionV, constraintSet, dhat, /*project_to_psd=*/true);
      // end for stuff to wrap.

      auto incrementalPotential = kineticPotential + h * h * elasticPotential + barrierPotential;
      auto incrementalPotentialGradient = kineticPotentialGradient + h * h * elasticPotentialGradient +
                                          barrierPotentialGradient;
      auto incrementalPotentialHessian = kineticPotentialHessian + h * h * elasticPotentialHessian +
                                         barrierPotentialHessian;
      double f = incrementalPotential;
      Eigen::VectorXd g = incrementalPotentialGradient;
      auto H_proj = incrementalPotentialHessian;

      // TODO need to add barrier to the function below.
      std::function<double(const Eigen::VectorXd &)> func = [&](const Eigen::VectorXd &x) {
        return kineticPotentialFunction(x) + h * h * elasticPotentialFunction(x);
      };

      Eigen::VectorXd d = cg_solver.compute(H_proj).solve(-g);
      Eigen::MatrixXd D(vertexCount, 3);
      elasticPotentialFunction.x_to_data(d, [&](int v_idx, const Eigen::Vector3d &v) { D.row(v_idx) = v; });

      D += vertexPositions;  // TODO: check whether this should be collisionV instead?
      double c = ipc::compute_collision_free_stepsize(collisionMesh, vertexPositions, D);
      // std::cout << "collision free step size: " << c << std::endl;

      if (TinyAD::newton_decrement(d, g) < convergence_eps) {
        // std::cout << "Final energy: " << func(x) << std::endl;
        // std::cout << "Decrement: " << TinyAD::newton_decrement(d, g) << std::endl;
        break;
      }
      // std::cout << "Energy: " << func(x) << std::endl;
      double s_max = min(c, 1.0);
      x = TinyAD::line_search(x, d, f, g, func, s_max);
    }
    positions = x;
    velocities = (x - x0).array() / dt;

    // Ugly code to reshape back to IGL format.
    Eigen::MatrixXd temp(vertexCount, 3);
    elasticPotentialFunction.x_to_data(
        positions, [&](int v_idx, const Eigen::Vector3d &vectorData) { temp.row(v_idx) = vectorData; });
    positionsHistory.push_back(temp);

    elasticPotentialFunction.x_to_data(
        velocities, [&](int v_idx, const Eigen::Vector3d &vectorData) { temp.row(v_idx) = vectorData; });
    velocitiesHistory.push_back(temp);

    elasticPotentialFunction.x_to_data(
        -elasticGradient, [&](int v_idx, const Eigen::Vector3d &vectorData) { temp.row(v_idx) = vectorData; });
    forcesHistory.push_back(temp);
  }

  map<string, vector<Eigen::MatrixXd>> vector3dVertexDataHistories;
  vector3dVertexDataHistories["velocities"] = velocitiesHistory;
  vector3dVertexDataHistories["forces"] = forcesHistory;

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 1.;
  polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{{-2., -2., -2.}, {2., 2., 2.}};

  std::cout << positionsHistory[0] << std::endl;

  // auto *psMesh = polyscope::registerSurfaceMesh("my mesh", positionsHistory[0], triangles);
  // psMesh->addVertexVectorQuantity("velocity", velocitiesHistory[10]);

  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());
  psMesh->addVertexParameterizationQuantity("restPosition", vertexRestPositions);
  polyscope::state::userCallback = [&]() -> void {
    callback(triangles, positionsHistory, vector3dVertexDataHistories);
  };
  polyscope::show();
}