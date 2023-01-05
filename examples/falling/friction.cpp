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

#include <igl/edges.h>
#include <igl/triangle/triangulate.h>

#include <ipc/friction/friction.hpp>
#include <ipc/ipc.hpp>

#include <Eigen/CholmodSupport>

using namespace std;
using namespace std::chrono;

tuple<VertexPositions, Triangles> makeTwoTriangleSquare() {
  Eigen::RowVector3d v0(-1.0, -1.0, 0.0);
  Eigen::RowVector3d v1(1.0, -1.0, 0.0);
  Eigen::RowVector3d v2(1.0, 1.0, 0.0);
  Eigen::RowVector3d v3(-1.0, 1.0, 0.0);

  Eigen::Matrix<double, 4, 3> vertexCoordinates(4, 3);
  vertexCoordinates << v0, v1, v2, v3;

  Eigen::RowVector3i triangle0{0, 1, 3};
  Eigen::RowVector3i triangle1{1, 2, 3};

  // With the triangles below, there is a parallel edge-edge collision, which results in a weird artefact.
  // Eigen::RowVector3i triangle0{0, 1, 2};
  // Eigen::RowVector3i triangle1{0, 2, 3};
  Eigen::ArrayX3i triangles(2, 3);
  triangles << triangle0, triangle1;

  return std::make_tuple(vertexCoordinates, triangles);
}

tuple<VertexPositions, Triangles> makeTriangle() {
  Eigen::RowVector3d v0(-1.0, -1.0, 0.0);
  Eigen::RowVector3d v1(1.0, -1.0, 0.0);
  Eigen::RowVector3d v2(1.0, 1.0, 0.0);

  Eigen::Matrix<double, 3, 3> vertexCoordinates(3, 3);
  vertexCoordinates << v0, v1, v2;

  Eigen::RowVector3i triangle0{0, 1, 2};
  Eigen::ArrayX3i triangles(1, 3);
  triangles << triangle0;

  return std::make_tuple(vertexCoordinates, triangles);
}

int main() {
  VertexPositions vertexPositions;
  vector<Points> pointGroups;
  vector<Edges> edgeGroups;
  vector<Triangles> triangleGroups;
  vector<Tetrahedra> tetrahedraGroups;

  // Add ground
  auto [groundVertices, groundTriangles] = makeTwoTriangleSquare();

  // 6 degress rotation
  Eigen::Matrix3d rotationMatrix = Eigen::AngleAxisd(M_PI / 30.0, Eigen::Vector3d::UnitX()).matrix();
  groundVertices = (rotationMatrix * groundVertices.transpose()).transpose();

  groundTriangles = silk::appendTriangles(vertexPositions, triangleGroups, groundVertices, groundTriangles);

  // Add cloth
  auto [clothVertices, clothTriangles] = makeTriangle();
  clothVertices.array() *= 0.2;
  clothVertices.col(2).array() += 0.1;
  clothTriangles = silk::appendTriangles(vertexPositions, triangleGroups, clothVertices, clothTriangles);

  std::cout << "cloth triangles: " << clothTriangles.rows() << std::endl;

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
      [](auto &&F) { return 2.0 * silk::baraffWitkinStretchPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);

  silk::TinyADEnergy stretchEnergy(triangleStretchScalarFunction);
  energies["triangleStretch"] = &stretchEnergy;

  // Adding the shear energy
  TinyAD::ScalarFunction<3, double, Eigen::Index> triangleShearScalarFunction = silk::createTriangleScalarFunction(
      [](auto &&F) { return 0.2 * silk::baraffWitkinShearPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);
  silk::TinyADEnergy shearEnergy(triangleShearScalarFunction);
  energies["triangleShear"] = &shearEnergy;

  int speedup = 1;
  int timesteps = 100;
  timesteps /= speedup;
  double timestepSize = 0.02;

  // vertexPositions(0, 0) += 0.1;  // Small deformation

  vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory;
  vertexPositionHistory.push_back(vertexPositions);

  Eigen::VectorXd positions = silk::flatten(vertexPositions);
  Eigen::VectorXd velocities = Eigen::VectorXd::Zero(positions.size());

  vector<map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>> vertexVectorQuantitiesHistory;

  double clothVertexMass = 1.0 / clothVertices.rows();
  Eigen::VectorXd vertexMasses = clothVertexMass * Eigen::VectorXd::Ones(vertexCount);
  vertexMasses.topRows(4).array() = 1.0;

  // Setting up the energy for the scripted vertices
  Points staticVertices(4);
  staticVertices << 0, 1, 2, 3;
  map<int, Eigen::Vector3d> scriptedPositions;
  for (int index : staticVertices) {
    scriptedPositions[index] = vertexPositions.row(index);
  }

  // Setting up the collision mesh
  Eigen::MatrixXi collisionTriangles(groundTriangles.rows() + clothTriangles.rows(), 3);
  collisionTriangles << groundTriangles, clothTriangles;
  Eigen::MatrixXi collisionEdges;
  igl::edges(collisionTriangles, collisionEdges);
  ipc::CollisionMesh collisionMesh(vertexPositions, collisionEdges, collisionTriangles);
  ipc::BroadPhaseMethod method = ipc::BroadPhaseMethod::HASH_GRID;
  ipc::Constraints contactConstraintSet;
  double dhat = 0.01;  // square of maximum distance at which repulsion works

  TinyAD::ScalarFunction<3, double, Eigen::Index> staticSpringEnergy = silk::createVertexEnergyFunction(
      vertexCount, staticVertices, scriptedPositions, vertexMasses);
  silk::TinyADEnergy staticSpring(staticSpringEnergy);
  energies["staticSpring"] = &staticSpring;

  double endTime = timesteps * timestepSize;

  auto start = high_resolution_clock::now();

  Eigen::MatrixXd laggedPositions = silk::unflatten(positions);

  for (int timestep = 0; timestep < timesteps; timestep++) {
    std::cout << "============== Timestep " << timestep << "=================" << std::endl;

    map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;

    double h = timestepSize;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;

    TinyAD::ScalarFunction<3, double, Eigen::Index> kineticPotentialFunction = silk::createKineticPotentialFunction(
        x0, v0, vertexMasses, h);
    silk::TinyADEnergy kineticPotential(kineticPotentialFunction);
    energies["kineticPotential"] = &kineticPotential;

    silk::IPCBarrierEnergy barrierEnergy(collisionMesh, contactConstraintSet, dhat);
    energies["contactBarrier"] = &barrierEnergy;

    // TODO: think about how to ensure/enforce all energies have a weight.
    double kappa = 1.0;

    auto x = x0;
    int maxNewtonIterations = 100;
    double convergenceAccuracy = 1e-5;

    // Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> linearSolver;
    Eigen::CholmodSimplicialLDLT<Eigen::SparseMatrix<double>> linearSolver;

    Eigen::VectorXd previousSearchDirection;

    std::cout << "Newton Iteration Start " << std::endl;
    for (int newtonIteration = 0; newtonIteration < maxNewtonIterations; ++newtonIteration) {
      // std::cout << "Newton iteration: " << newtonIteration << std::endl;

      silk::IPCFrictionEnergy frictionEnergy(collisionMesh, contactConstraintSet, dhat, laggedPositions, h);
      energies["friction"] = &frictionEnergy;

      if (newtonIteration == 0) {
        // Store energies for visualization
        for (auto const &[name, energy_ptr] : energies) {
          silk::Energy &energy = *energy_ptr;
          auto [f, g, H] = energy.eval_with_hessian_proj(x0);
          cout << name << ": " << f << endl;
          vertexVectorQuantities[name] = silk::unflatten(-g);
        }
      }

      map<string, double> energyWeights = {{"kineticPotential", 1.0},
                                           {"triangleStretch", h * h},
                                           {"triangleShear", h * h},
                                           {"staticSpring", h * h},
                                           {"contactBarrier", kappa},
                                           {"friction", h * h}};
      silk::AdditiveEnergy incrementalPotential(energies, energyWeights);

      std::function<double(const Eigen::VectorXd &)> func = [&](const Eigen::VectorXd &x) {
        return incrementalPotential(x);
      };

      auto [f, g, H_proj] = incrementalPotential.eval_with_hessian_proj(x);
      Eigen::VectorXd d = linearSolver.compute(H_proj).solve(-g);

      // cosine of the angle between the search direction and the previous search direction
      // if (newtonIteration > 0) {
      //   double cosine = d.dot(previousSearchDirection) / (d.norm() * previousSearchDirection.norm());
      //   std::cout << "Cosine: " << cosine << std::endl;
      // }
      // previousSearchDirection = d;

      // auto [f, g] = incrementalPotential.eval_with_gradient(x);
      // Eigen::VectorXd d = -g;  // Do gradient descent with line search

      double stoppingMeasure = d.cwiseAbs().maxCoeff() / h;
      std::cout << newtonIteration << ": " << stoppingMeasure << std::endl;

      // std::cout << "Stopping measure: " << d.cwiseAbs().maxCoeff() / h << std::endl;

      if (silk::convergenceConditionIPC(d, h, 0.0001)) {
        break;
      }

      Eigen::MatrixXd direction = silk::unflatten(d);
      Eigen::MatrixXd positions_ = silk::unflatten(x);
      // TODO: only update constraint set once (already done in IPCBarrierEnergy)
      Eigen::MatrixXd collisionV = collisionMesh.vertices(positions_);
      contactConstraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);

      double c = ipc::compute_collision_free_stepsize(collisionMesh, positions_, positions_ + direction, method, dhat);
      double maxStepSize = min(c, 1.0);

      // Eigen::MatrixXd laggedV = collisionMesh.vertices(laggedPositions);

      laggedPositions = positions_;  // lagging inside newton iterations

      // std::cout << "Friction: " << friction << std::endl;

      x = silk::backtrackingLineSearch(x, d, f, g, func, maxStepSize);
    }

    // laggedPositions = silk::unflatten(x);

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