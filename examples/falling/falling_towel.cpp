#include "silk/conversions.hh"
#include "silk/energies.hh"
#include "silk/visualization.hh"

#include <iostream>

#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>

#include <igl/edges.h>
#include <igl/triangle/triangulate.h>

#include <ipc/ipc.hpp>

using namespace std;

void callback(vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory,
              vector<Eigen::ArrayXi> pointGroups,
              vector<Eigen::ArrayX2i> edgeGroups,
              vector<Eigen::ArrayX3i> triangleGroups,
              vector<Eigen::ArrayX4i> tetrahedraGroups,
              vector<map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>> vertexVectorQuantitiesHistory) {

  int frame = silk::state::playback_frame_counter % vertexPositionHistory.size();
  ImGui::Text("Frame %d", frame);

  if (ImGui ::Button("Next frame")) {
    silk::state::playback_frame_counter++;
    frame = silk::state::playback_frame_counter % vertexPositionHistory.size();
    silk::registerInPolyscope(vertexPositionHistory[frame],
                              pointGroups,
                              edgeGroups,
                              triangleGroups,
                              tetrahedraGroups,
                              vertexVectorQuantitiesHistory[frame]);
  }

  if (ImGui ::Button("Previous frame")) {
    silk::state::playback_frame_counter--;
    frame = silk::state::playback_frame_counter % vertexPositionHistory.size();
    silk::registerInPolyscope(vertexPositionHistory[frame],
                              pointGroups,
                              edgeGroups,
                              triangleGroups,
                              tetrahedraGroups,
                              vertexVectorQuantitiesHistory[frame]);
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

  silk::registerInPolyscope(vertexPositionHistory[frame],
                            pointGroups,
                            edgeGroups,
                            triangleGroups,
                            tetrahedraGroups,
                            vertexVectorQuantitiesHistory[frame]);
  silk::state::playback_frame_counter++;
}

tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::ArrayX3i> makeTriangulatedSquare() {
  Eigen::RowVector3d v0(-1.0, -1.0, 0.0);
  Eigen::RowVector3d v1(1.0, -1.0, 0.0);
  Eigen::RowVector3d v2(1.0, 1.0, 0.0);
  Eigen::RowVector3d v3(-1.0, 1.0, 0.0);

  Eigen::Matrix<double, 4, 3> vertexCoordinates(4, 3);
  vertexCoordinates << v0, v1, v2, v3;

  Eigen::MatrixXi edges(4, 2);  // Mesh edges
  edges << 0, 1, 1, 2, 2, 3, 3, 0;

  Eigen::MatrixXd vertexCoordinates2D = vertexCoordinates.leftCols(2);

  Eigen::MatrixXd newVertices2D;
  Eigen::MatrixXi newTriangles;

  Eigen::MatrixXd H;

  igl::triangle::triangulate(vertexCoordinates2D, edges, H, "a0.005q", newVertices2D, newTriangles);

  Eigen::MatrixXd newVertexCoordinates = Eigen::MatrixXd::Zero(newVertices2D.rows(), 3);
  newVertexCoordinates.leftCols(2) = newVertices2D;

  return std::make_tuple(newVertexCoordinates, newTriangles);
}

tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::ArrayX3i> makeTwoTriangleSquare() {
  Eigen::RowVector3d v0(-1.0, -1.0, 0.0);
  Eigen::RowVector3d v1(1.0, -1.0, 0.0);
  Eigen::RowVector3d v2(1.0, 1.0, 0.0);
  Eigen::RowVector3d v3(-1.0, 1.0, 0.0);

  Eigen::Matrix<double, 4, 3> vertexCoordinates(4, 3);
  vertexCoordinates << v0, v1, v2, v3;

  Eigen::RowVector3i triangle0{0, 1, 2};
  Eigen::RowVector3i triangle1{0, 2, 3};
  Eigen::ArrayX3i triangles(2, 3);
  triangles << triangle0, triangle1;

  return std::make_tuple(vertexCoordinates, triangles);
}

tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::ArrayX3i> appendElements(
    Eigen::Matrix<double, Eigen::Dynamic, 3> &vertices,
    Eigen::Matrix<double, Eigen::Dynamic, 3> &newVertices,
    Eigen::ArrayX3i &newElements) {
  newElements += vertices.rows();
  vertices.conservativeResize(vertices.rows() + newVertices.rows(), 3);
  vertices.bottomRows(newVertices.rows()) = newVertices;
  return std::make_tuple(vertices, newElements);
}

int main() {

  Eigen::Matrix<double, Eigen::Dynamic, 3> vertexPositions;

  Eigen::Matrix<double, Eigen::Dynamic, 3> groundVertices;
  Eigen::ArrayX3i groundTriangles;
  std::tie(groundVertices, groundTriangles) = makeTwoTriangleSquare();
  std::tie(vertexPositions, groundTriangles) = appendElements(vertexPositions, groundVertices, groundTriangles);

  Eigen::Matrix<double, Eigen::Dynamic, 3> clothVertices;
  Eigen::ArrayX3i clothTriangles;
  std::tie(clothVertices, clothTriangles) = makeTriangulatedSquare();
  Eigen::Vector3d unitXY = (Eigen::Vector3d::UnitX() + Eigen::Vector3d::UnitY()).normalized();
  Eigen::Matrix3d rotationMatrix = Eigen::AngleAxisd(M_PI / 2.0, unitXY).matrix();
  clothVertices = (rotationMatrix * clothVertices.transpose()).transpose();
  clothVertices.array() *= 0.5;
  clothVertices.col(2).array() += 0.8;
  std::tie(vertexPositions, clothTriangles) = appendElements(vertexPositions, clothVertices, clothTriangles);

  vector<Eigen::ArrayXi> pointGroups;
  vector<Eigen::ArrayX2i> edgeGroups;
  vector<Eigen::ArrayX3i> triangleGroups{groundTriangles, clothTriangles};
  vector<Eigen::ArrayX4i> tetrahedraGroups;

  Eigen::MatrixXi collisionTriangles(groundTriangles.rows() + clothTriangles.rows(), 3);
  collisionTriangles << groundTriangles, clothTriangles;

  Eigen::MatrixXi collisionEdges;
  igl::edges(collisionTriangles, collisionEdges);

  std::cout << vertexPositions << std::endl;

  vector<Eigen::Matrix2d> invertedTriangleRestShapes = silk::initializeInvertedTriangleRestShapes(vertexPositions,
                                                                                                  clothTriangles);
  Eigen::VectorXd triangleRestAreas = silk::calculateAreas(vertexPositions, clothTriangles);

  // From this point, no more vertices can be added. TinyAD needs this number at compile time.
  int vertexCount = vertexPositions.rows();

  auto triangleStretchPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));

  triangleStretchPotentialFunction.add_elements<3>(
      TinyAD::range(clothTriangles.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
        using ScalarT = TINYAD_SCALAR_TYPE(element);
        int triangleIndex = element.handle;
        Eigen::Matrix2d invertedRestShape = invertedTriangleRestShapes[triangleIndex];

        Eigen::Vector3<ScalarT> x0 = element.variables(clothTriangles(triangleIndex, 0));
        Eigen::Vector3<ScalarT> x1 = element.variables(clothTriangles(triangleIndex, 1));
        Eigen::Vector3<ScalarT> x2 = element.variables(clothTriangles(triangleIndex, 2));

        Eigen::Matrix<ScalarT, 3, 2> F = silk::triangleDeformationGradient(x0, x1, x2, invertedRestShape);
        ScalarT stretchPotential = silk::baraffWitkinStretchPotential(F);

        double stretchStiffness = 1.0;
        double restArea = triangleRestAreas(triangleIndex);
        double areaFactor = sqrt(restArea);

        ScalarT stretchEnergy = stretchStiffness * areaFactor * stretchPotential;

        return stretchEnergy;
      });

  int timesteps = 400;
  double timestepSize = 0.001;

  vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory;
  // vertexPositions(3, 2) += 0.5;
  vertexPositionHistory.push_back(vertexPositions);
  Eigen::VectorXd initialPositions = silk::flatten(vertexPositions);
  Eigen::VectorXd initialVelocities = Eigen::VectorXd::Zero(initialPositions.size());

  vector<map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>> vertexVectorQuantitiesHistory;
  // auto [stretch, stretchGradient] = triangleStretchPotentialFunction.eval_with_gradient(initialPositions);
  // vertexVectorQuantities["triangleStretchForces"] = silk::unflatten(-stretchGradient);

  Eigen::VectorXd positions = initialPositions;
  Eigen::VectorXd velocities = initialVelocities;

  // Set up gravity
  double standard_gravity = -9.81; /* in m/s^2 */

  Eigen::MatrixXd gravityAccelerations = Eigen::MatrixXd::Zero(vertexCount, 3);
  gravityAccelerations.col(2) = standard_gravity * Eigen::VectorXd::Ones(vertexCount);
  // gravityAccelerations.topRows(3).col(2).array() = 0.0;

  Eigen::VectorXd gravityAccelerationsFlat = silk::flatten(gravityAccelerations);

  // Eigen::VectorXd gravityAccelerations = Eigen::VectorXd::Zero(3 * vertexCount);

  // Eigen::VectorXd gravityAccelerations = Eigen::VectorXd::Zero(3 * vertexCount);
  // gravityAccelerations(Eigen::seqN(2, vertexCount, 3)) = standard_gravity * Eigen::VectorXd::Ones(vertexCount);

  double vertexMass = 1.0 / vertexCount;
  Eigen::VectorXd vertexMasses = vertexMass * Eigen::VectorXd::Ones(vertexCount);

  // Eigen::VectorXd gravityForces = vertexMasses * gravityAccelerations;
  // Eigen::VectorXd externalForces = Eigen::VectorXd::Zero(positions.size());

  // std::cout << gravityAccelerations << std::endl;

  // Script vertex 0, 1 and 2 to their initial positions
  Eigen::ArrayXi scriptedVertices(4);
  scriptedVertices << 0, 1, 2, 3;
  map<int, Eigen::Vector3d> scriptedPositions;
  for (int i = 0; i < scriptedVertices.size(); i++) {
    int vertexIndex = scriptedVertices(i);
    scriptedPositions[vertexIndex] = vertexPositions.row(vertexIndex);
  }

  ipc::CollisionMesh collisionMesh(vertexPositions, collisionEdges, collisionTriangles);
  ipc::BroadPhaseMethod method = ipc::BroadPhaseMethod::BRUTE_FORCE;
  ipc::Constraints constraintSet;
  double dhat = 0.01;  // square of maximum distance at which repulsion works

  for (int i = 0; i < timesteps; i++) {

    // for (int i = 0; i < scriptedVertices.size(); i++) {
    //   int vertexIndex = scriptedVertices(i);
    //   scriptedPositions[vertexIndex](2) += timestepSize;
    // }

    map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;

    std::cout << "Timestep: " << i << std::endl;

    double h = timestepSize;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;
    // Eigen::VectorXd fExt = externalForces;

    Eigen::VectorXd predictivePositionsFlat = x0 + h * v0 + h * h * gravityAccelerationsFlat;
    Eigen::MatrixXd predictivePositions = silk::unflatten(predictivePositionsFlat);

    auto x = x0;
    int maxNewtonIterations = 50;
    double convergence_eps = 1e-20;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> conjugateGradientSolver;
    for (int j = 0; j < maxNewtonIterations; ++j) {
      std::cout << "Newton iteration: " << j << std::endl;

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

      auto scriptedPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));
      scriptedPotentialFunction.add_elements<1>(
          TinyAD::range(scriptedVertices.size()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
            using ScalarT = TINYAD_SCALAR_TYPE(element);
            int vertexIndex = element.handle;
            Eigen::Vector3<ScalarT> position = element.variables(vertexIndex);
            Eigen::Vector3d scriptedPosition = scriptedPositions[vertexIndex];
            Eigen::Vector3<ScalarT> difference = position - scriptedPosition;

            double vertexMass = vertexMasses(vertexIndex);

            ScalarT potential = 0.5 * 1000000.0 * vertexMass * difference.transpose() * difference;
            return potential;
          });

      auto [triangleStretchPotential,
            triangleStretchPotentialGradient,
            triangleStretchPotentialHessian] = triangleStretchPotentialFunction.eval_with_hessian_proj(x);

      auto [kineticPotential,
            kineticPotentialGradient,
            kineticPotentialHessian] = kineticPotentialFunction.eval_with_hessian_proj(x);

      auto [scriptedPotential,
            scriptedPotentialGradient,
            scriptedPotentialHessian] = scriptedPotentialFunction.eval_with_hessian_proj(x);

      // Barrier potential, gradient and hessian are provide by the ipc-toolkit
      Eigen::MatrixXd collisionV = collisionMesh.vertices(silk::unflatten(x));
      constraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);
      double barrierPotential = ipc::compute_barrier_potential(collisionMesh, collisionV, constraintSet, dhat);

      Eigen::VectorXd barrierPotentialGradient = ipc::compute_barrier_potential_gradient(
          collisionMesh, collisionV, constraintSet, dhat);
      Eigen::SparseMatrix<double> barrierPotentialHessian = ipc::compute_barrier_potential_hessian(
          collisionMesh, collisionV, constraintSet, dhat, /*project_to_psd=*/true);

      if (j == 0) {
        vertexVectorQuantities["triangleStretchForces"] = silk::unflatten(-triangleStretchPotentialGradient);
        vertexVectorQuantities["kineticGradient"] = silk::unflatten(-kineticPotentialGradient);
        vertexVectorQuantities["scriptedGradient"] = silk::unflatten(-scriptedPotentialGradient);
        vertexVectorQuantities["barrierGradient"] = silk::unflatten(-barrierPotentialGradient);
      }

      // std::cout << "kineticPotential: " << kineticPotential << std::endl;
      // std::cout << kineticPotentialGradient << std::endl;

      auto incrementalPotential = kineticPotential + scriptedPotential + barrierPotential +
                                  h * h * triangleStretchPotential;
      auto incrementalPotentialGradient = kineticPotentialGradient + scriptedPotentialGradient +
                                          barrierPotentialGradient + h * h * triangleStretchPotentialGradient;
      auto incrementalPotentialHessian = kineticPotentialHessian + scriptedPotentialHessian + barrierPotentialHessian +
                                         h * h * triangleStretchPotentialHessian;
      double f = incrementalPotential;
      Eigen::VectorXd g = incrementalPotentialGradient;
      auto H_proj = incrementalPotentialHessian;

      std::function<double(const Eigen::VectorXd &)> func = [&](const Eigen::VectorXd &x) {
        Eigen::MatrixXd collisionV = collisionMesh.vertices(silk::unflatten(x));
        constraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);
        double barrierPotential = ipc::compute_barrier_potential(collisionMesh, collisionV, constraintSet, dhat);

        return kineticPotentialFunction(x) + scriptedPotentialFunction(x) + barrierPotential +
               h * h * triangleStretchPotentialFunction(x);
      };

      Eigen::VectorXd d = conjugateGradientSolver.compute(H_proj).solve(-g);

      Eigen::MatrixXd direction = silk::unflatten(d);
      Eigen::MatrixXd positions_ = silk::unflatten(x);
      double c = ipc::compute_collision_free_stepsize(collisionMesh, positions_, positions_ + direction);

      if (TinyAD::newton_decrement(d, g) < convergence_eps) {
        break;
      }
      double s_max = min(c, 1.0);
      x = TinyAD::line_search(x, d, f, g, func, s_max);
    }

    positions = x;
    velocities = (x - x0).array() / h;

    vertexPositionHistory.push_back(silk::unflatten(positions));

    vertexVectorQuantities["velocities"] = silk::unflatten(velocities);
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
    callback(vertexPositionHistory,
             pointGroups,
             edgeGroups,
             triangleGroups,
             tetrahedraGroups,
             vertexVectorQuantitiesHistory);
  };

  polyscope::show();
}