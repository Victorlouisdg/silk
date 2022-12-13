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
              vector<Eigen::ArrayX4i> tetrahedraGroups) {

  int frame = silk::state::playback_frame_counter % vertexPositionHistory.size();
  ImGui::Text("Frame %d", frame);

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

  silk::registerInPolyscope(vertexPositionHistory[frame], pointGroups, edgeGroups, triangleGroups, tetrahedraGroups);
  silk::state::playback_frame_counter++;
}

int main() {
  Eigen::Matrix<double, Eigen::Dynamic, 3> vertexPositions;

  // Add the spring
  Eigen::RowVector3d v0(0.0, 0.0, 3.0);
  Eigen::RowVector3d v1(0.0, 0.0, 2.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v0, v1;
  Eigen::ArrayX2i edges(1, 2);
  edges << 0, 1;

  // Add the triangle
  Eigen::RowVector3d v2(1.0, 0.0, 1.0);
  Eigen::RowVector3d v3(-1.0, 0.0, 1.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v2, v3;
  Eigen::ArrayX3i triangles(1, 3);
  triangles << 1, 2, 3;

  // Add the tetrahedron
  Eigen::RowVector3d v4(0.0, 1.0, 0.0);
  Eigen::RowVector3d v5(0.0, -1.0, 0.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v4, v5;
  Eigen::ArrayX4i tetrahedra(1, 4);
  tetrahedra << 2, 3, 4, 5;

  // Add an extra point
  Eigen::RowVector3d v6(0.0, 0.0, 4.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 1, 3);
  vertexPositions.bottomRows(1) << v6;
  Eigen::ArrayXi points(3);
  points << 0, 1, 6;

  vector<Eigen::ArrayXi> pointGroups{points};
  vector<Eigen::ArrayX2i> edgeGroups{edges};
  vector<Eigen::ArrayX3i> triangleGroups{triangles};
  vector<Eigen::ArrayX4i> tetrahedraGroups{tetrahedra};

  std::cout << vertexPositions << std::endl;

  // TODO: set up rest shapes
  Eigen::VectorXd springRestLengths(edges.rows());
  springRestLengths(0) = (v0 - v1).norm();

  vector<Eigen::Matrix2d> invertedTriangleRestShapes = silk::initializeInvertedTriangleRestShapes(vertexPositions,
                                                                                                  triangles);
  Eigen::VectorXd triangleRestAreas = silk::calculateAreas(vertexPositions, triangles);

  vector<Eigen::Matrix3d> invertedTetrahedronRestShapes = silk::initializeInvertedTetrahedronRestShapes(
      vertexPositions, tetrahedra);

  // From this point, no more vertices can be added. TinyAD needs this number at compile time.
  int vertexCount = vertexPositions.rows();

  // Set up a function with 3D vertex positions as variables
  auto tetrahedronElasticPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));

  tetrahedronElasticPotentialFunction.add_elements<4>(
      TinyAD::range(tetrahedra.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
        using ScalarT = TINYAD_SCALAR_TYPE(element);
        int tetrahedronIndex = element.handle;
        Eigen::Matrix3d Mr_inv = invertedTetrahedronRestShapes[tetrahedronIndex];

        Eigen::Vector3<ScalarT> x0 = element.variables(tetrahedra(tetrahedronIndex, 0));
        Eigen::Vector3<ScalarT> x1 = element.variables(tetrahedra(tetrahedronIndex, 1));
        Eigen::Vector3<ScalarT> x2 = element.variables(tetrahedra(tetrahedronIndex, 2));
        Eigen::Vector3<ScalarT> x3 = element.variables(tetrahedra(tetrahedronIndex, 3));

        Eigen::Matrix3<ScalarT> M;
        M << x1 - x0, x2 - x0, x3 - x0;

        // Compute tet Jacobian and energy
        // Eigen::Matrix3d Mr = restShapes[t_idx];
        Eigen::Matrix3<ScalarT> J = M * Mr_inv;
        // double vol = Mr.determinant() / 6.0;
        double restVolume = 1.0;

        // The exponential symmetric Dirichlet energy:
        return restVolume * exp((J.squaredNorm() + J.inverse().squaredNorm()));
      });

  auto triangleStretchPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));

  triangleStretchPotentialFunction.add_elements<3>(
      TinyAD::range(triangles.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
        using ScalarT = TINYAD_SCALAR_TYPE(element);
        int triangleIndex = element.handle;
        Eigen::Matrix2d invertedRestShape = invertedTriangleRestShapes[triangleIndex];

        Eigen::Vector3<ScalarT> x0 = element.variables(triangles(triangleIndex, 0));
        Eigen::Vector3<ScalarT> x1 = element.variables(triangles(triangleIndex, 1));
        Eigen::Vector3<ScalarT> x2 = element.variables(triangles(triangleIndex, 2));

        Eigen::Matrix<ScalarT, 3, 2> F = silk::triangleDeformationGradient(x0, x1, x2, invertedRestShape);
        ScalarT stretchPotential = silk::baraffWitkinStretchPotential(F);

        double stretchStiffness = 1.0;
        double restArea = triangleRestAreas(triangleIndex);
        double areaFactor = sqrt(restArea);

        ScalarT stretchEnergy = stretchStiffness * areaFactor * stretchPotential;

        // The exponential symmetric Dirichlet energy:
        return stretchEnergy;
      });

  double vertexMass = 1.0 / vertexCount;
  Eigen::VectorXd vertexMasses = vertexMass * Eigen::VectorXd::Ones(vertexCount);

  Eigen::VectorXd initialPositions = silk::flatten(vertexPositions);
  initialPositions(3 * 4 + 2) -= 0.5;  // perturb the tetrahedron

  auto [elastic, elasticGradient] = tetrahedronElasticPotentialFunction.eval_with_gradient(initialPositions);
  auto [stretch, stretchGradient] = triangleStretchPotentialFunction.eval_with_gradient(initialPositions);

  map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;
  vertexVectorQuantities["tetrahedronElasticForces"] = silk::unflatten(-elasticGradient);
  vertexVectorQuantities["triangleStretchForces"] = silk::unflatten(-stretchGradient);

  vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory;
  vertexPositionHistory.push_back(vertexPositions);
  int timesteps = 500;  // 100;
  double timestepSize = 0.001;

  Eigen::VectorXd initialVelocities = Eigen::VectorXd::Zero(initialPositions.size());

  // For the simulation part, we mostly work with flat (3*n_vertices, 1) column vectors.
  Eigen::VectorXd positions = initialPositions;
  Eigen::VectorXd velocities = initialVelocities;

  for (int i = 0; i < timesteps; i++) {

    std::cout << i << std::endl;

    double h = timestepSize;
    Eigen::VectorXd x0 = positions;
    Eigen::VectorXd v0 = velocities;

    Eigen::VectorXd predictivePositionsFlat = x0 + h * v0;  //+ h * h;  // * Minv
    Eigen::MatrixXd predictivePositions = silk::unflatten(predictivePositionsFlat);

    auto x = x0;
    int maxNewtonIterations = 50;
    double convergence_eps = 1e-8;
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> conjugateGradientSolver;
    for (int i = 0; i < maxNewtonIterations; ++i) {

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

      auto [tetrahedronElasticPotential,
            tetrahedronElasticPotentialGradient,
            tetrahedronElasticPotentialHessian] = tetrahedronElasticPotentialFunction.eval_with_hessian_proj(x);

      auto [triangleStretchPotential,
            triangleStretchPotentialGradient,
            triangleStretchPotentialHessian] = triangleStretchPotentialFunction.eval_with_hessian_proj(x);

      auto [kineticPotential,
            kineticPotentialGradient,
            kineticPotentialHessian] = kineticPotentialFunction.eval_with_hessian_proj(x);

      auto incrementalPotential = kineticPotential + h * h * (tetrahedronElasticPotential + triangleStretchPotential);
      auto incrementalPotentialGradient = kineticPotentialGradient +
                                          h * h *
                                              (tetrahedronElasticPotentialGradient + triangleStretchPotentialGradient);
      auto incrementalPotentialHessian = kineticPotentialHessian +
                                         h * h *
                                             (tetrahedronElasticPotentialHessian + triangleStretchPotentialHessian);
      double f = incrementalPotential;
      Eigen::VectorXd g = incrementalPotentialGradient;
      auto H_proj = incrementalPotentialHessian;

      std::function<double(const Eigen::VectorXd &)> func = [&](const Eigen::VectorXd &x) {
        return kineticPotentialFunction(x) +
               h * h * (tetrahedronElasticPotentialFunction(x) + triangleStretchPotentialFunction(x));
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
  }

  Eigen::Matrix<double, Eigen::Dynamic, 3> perturbedVertexPositions = silk::unflatten(initialPositions);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 5.;
  // polyscope::state::boundingBox = std::tuple<glm::vec3, glm::vec3>{{-2., -2., -2.}, {2., 2., 2.}};

  polyscope::state::userCallback = [&]() -> void {
    callback(vertexPositionHistory, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups);
  };
  // silk::registerInPolyscope(
  //     perturbedVertexPositions, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups, vertexVectorQuantities);
  polyscope::show();
}