#include "silk/conversions.hh"
#include "silk/energies.hh"
#include "silk/visualization.hh"

#include <iostream>

#include <TinyAD/Utils/LineSearch.hh>
#include <TinyAD/Utils/NewtonDecrement.hh>

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

tuple<Eigen::Matrix<double, Eigen::Dynamic, 3>, Eigen::ArrayX3i> makeOrthogonalTriangles() {
  // Setup mesh and rest shape

  // Set v0, v1 and v2 to the vertices of a regular triangle
  double a = 1.0 / 3.0;
  double b = sqrt(8.0 / 9.0);
  double c = sqrt(2.0 / 9.0);
  double d = sqrt(2.0 / 3.0);

  Eigen::RowVector3d v0(b, 0, -a);
  Eigen::RowVector3d v1(-c, d, -a);
  Eigen::RowVector3d v2(-c, -d, -a);

  Eigen::RowVector3d v3 = v0;
  Eigen::RowVector3d v4 = v1;
  Eigen::RowVector3d v5 = v2;

  Eigen::Matrix<double, 6, 3> vertexCoordinates;
  vertexCoordinates << v0, v1, v2, v3, v4, v5;

  Eigen::Matrix3d RotationMatrixY = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitY()).matrix();

  // Multiply the bottom 3 rows of the matrix by the rotation matrix
  vertexCoordinates.bottomRows(3) = (RotationMatrixY * vertexCoordinates.bottomRows(3).transpose()).transpose();

  // Shift the bottom 3 triangles up by 1.2
  vertexCoordinates.bottomRows(3).col(2).array() += 1.2;

  // Shift the vertices of the second triangle along the x-axis so that it is centered above the first
  vertexCoordinates.bottomRows(3).col(0).array() = (1. / 3. * b - 2. / 3. * c);

  Eigen::RowVector3i triangle0{0, 1, 2};
  Eigen::RowVector3i triangle1{3, 4, 5};
  Eigen::Array<int, 2, 3> triangles;
  // Eigen::ArrayX3i triangles;
  triangles << triangle0, triangle1;

  return std::make_tuple(vertexCoordinates, triangles);
};

int main() {

  Eigen::Matrix<double, Eigen::Dynamic, 3> vertexPositions;
  Eigen::ArrayX3i twoTriangles;

  std::tie(vertexPositions, twoTriangles) = makeOrthogonalTriangles();

  Eigen::ArrayX3i triangleFixed = twoTriangles.row(0);
  Eigen::ArrayX3i triangleElastic = twoTriangles.row(1);

  // TODO consider making these map<string, Array> so names can be used and shown in Polyscope.
  vector<Eigen::ArrayXi> pointGroups;
  vector<Eigen::ArrayX2i> edgeGroups;
  vector<Eigen::ArrayX3i> triangleGroups{triangleFixed, triangleElastic};
  vector<Eigen::ArrayX4i> tetrahedraGroups;

  std::cout << vertexPositions << std::endl;

  vector<Eigen::Matrix2d> invertedTriangleRestShapes = silk::initializeInvertedTriangleRestShapes(vertexPositions,
                                                                                                  triangleElastic);
  Eigen::VectorXd triangleRestAreas = silk::calculateAreas(vertexPositions, triangleElastic);

  // From this point, no more vertices can be added. TinyAD needs this number at compile time.
  int vertexCount = vertexPositions.rows();

  auto triangleStretchPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));

  triangleStretchPotentialFunction.add_elements<3>(
      TinyAD::range(triangleElastic.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
        using ScalarT = TINYAD_SCALAR_TYPE(element);
        int triangleIndex = element.handle;
        Eigen::Matrix2d invertedRestShape = invertedTriangleRestShapes[triangleIndex];

        Eigen::Vector3<ScalarT> x0 = element.variables(triangleElastic(triangleIndex, 0));
        Eigen::Vector3<ScalarT> x1 = element.variables(triangleElastic(triangleIndex, 1));
        Eigen::Vector3<ScalarT> x2 = element.variables(triangleElastic(triangleIndex, 2));

        Eigen::Matrix<ScalarT, 3, 2> F = silk::triangleDeformationGradient(x0, x1, x2, invertedRestShape);
        ScalarT stretchPotential = silk::baraffWitkinStretchPotential(F);

        double stretchStiffness = 1.0;
        double restArea = triangleRestAreas(triangleIndex);
        double areaFactor = sqrt(restArea);

        ScalarT stretchEnergy = stretchStiffness * areaFactor * stretchPotential;

        return stretchEnergy;
      });

  int timesteps = 200;
  double timestepSize = 0.001;

  vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory;
  vertexPositions(3, 2) += 0.5;
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
  gravityAccelerations.topRows(3).col(2).array() = 0.0;

  Eigen::VectorXd gravityAccelerationsFlat = silk::flatten(gravityAccelerations);

  // Eigen::VectorXd gravityAccelerations = Eigen::VectorXd::Zero(3 * vertexCount);

  // Eigen::VectorXd gravityAccelerations = Eigen::VectorXd::Zero(3 * vertexCount);
  // gravityAccelerations(Eigen::seqN(2, vertexCount, 3)) = standard_gravity * Eigen::VectorXd::Ones(vertexCount);

  double vertexMass = 1.0 / vertexCount;
  Eigen::VectorXd vertexMasses = vertexMass * Eigen::VectorXd::Ones(vertexCount);

  // Eigen::VectorXd gravityForces = vertexMasses * gravityAccelerations;
  // Eigen::VectorXd externalForces = Eigen::VectorXd::Zero(positions.size());

  // std::cout << gravityAccelerations << std::endl;

  for (int i = 0; i < timesteps; i++) {

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

      auto [triangleStretchPotential,
            triangleStretchPotentialGradient,
            triangleStretchPotentialHessian] = triangleStretchPotentialFunction.eval_with_hessian_proj(x);

      auto [kineticPotential,
            kineticPotentialGradient,
            kineticPotentialHessian] = kineticPotentialFunction.eval_with_hessian_proj(x);

      if (j == 0) {
        vertexVectorQuantities["triangleStretchForces"] = silk::unflatten(-triangleStretchPotentialGradient);
        vertexVectorQuantities["kineticGradient"] = silk::unflatten(-kineticPotentialGradient);
      }

      // std::cout << "kineticPotential: " << kineticPotential << std::endl;
      // std::cout << kineticPotentialGradient << std::endl;

      auto incrementalPotential = kineticPotential + h * h * triangleStretchPotential;
      auto incrementalPotentialGradient = kineticPotentialGradient + h * h * triangleStretchPotentialGradient;
      auto incrementalPotentialHessian = kineticPotentialHessian + h * h * triangleStretchPotentialHessian;
      double f = incrementalPotential;
      Eigen::VectorXd g = incrementalPotentialGradient;
      auto H_proj = incrementalPotentialHessian;

      std::function<double(const Eigen::VectorXd &)> func = [&](const Eigen::VectorXd &x) {
        return kineticPotentialFunction(x) + h * h * triangleStretchPotentialFunction(x);
      };

      Eigen::VectorXd d = conjugateGradientSolver.compute(H_proj).solve(-g);

      if (TinyAD::newton_decrement(d, g) < convergence_eps) {
        break;
      }
      x = TinyAD::line_search(x, d, f, g, func);
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