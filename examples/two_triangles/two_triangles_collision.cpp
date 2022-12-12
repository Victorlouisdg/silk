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

#include "silk/IGLGeometryCentralConvert.hh"

#include "silk/energies.hh"
#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

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

    auto [energy, gradient, projectedHessian] = elasticPotentialFunction.eval_with_hessian_proj(positions);
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

    // Ugly code to reshape back to IGL format.
    Eigen::MatrixXd temp(vertexCount, 3);
    elasticPotentialFunction.x_to_data(
        positions, [&](int v_idx, const Eigen::Vector3d &vectorData) { temp.row(v_idx) = vectorData; });
    positionsHistory.push_back(temp);

    elasticPotentialFunction.x_to_data(
        velocities, [&](int v_idx, const Eigen::Vector3d &vectorData) { temp.row(v_idx) = vectorData; });
    velocitiesHistory.push_back(temp);

    elasticPotentialFunction.x_to_data(
        forces, [&](int v_idx, const Eigen::Vector3d &vectorData) { temp.row(v_idx) = vectorData; });
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