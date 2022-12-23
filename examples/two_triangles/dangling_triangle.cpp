#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "silk/conversions.hh"
#include "silk/deformation/rest_shapes.hh"
#include "silk/energies.hh"
#include "silk/geometry/area.hh"
#include "silk/mesh_construction.hh"
#include "silk/simple_meshes.hh"
#include "silk/types.hh"
#include "silk/visualization.hh"

#include <igl/triangle/triangulate.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

PotentialFunction makePotentialFunction(int vertexCount,
                                        const Triangles &triangles,
                                        const vector<Eigen::Matrix2d> &triangleInvertedRestShapes,
                                        const Eigen::VectorXd &triangleRestAreas,
                                        const Eigen::VectorXd &stiffnesses) {

  PotentialFunction potentialFunction = [=](const Eigen::VectorXd &x) {
    auto tinyadFunction = TinyAD::scalar_function<3>(TinyAD::range(vertexCount));

    std::cout << "vertexCount:" << vertexCount << std::endl;
    std::cout << triangles.rows() << std::endl;

    tinyadFunction.add_elements<3>(TinyAD::range(triangles.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
      using ScalarT = TINYAD_SCALAR_TYPE(element);
      int triangleIndex = element.handle;
      Eigen::Matrix2d invertedRestShape = triangleInvertedRestShapes[triangleIndex];
      Eigen::Vector3<ScalarT> x0 = element.variables(triangles(triangleIndex, 0));
      Eigen::Vector3<ScalarT> x1 = element.variables(triangles(triangleIndex, 1));
      Eigen::Vector3<ScalarT> x2 = element.variables(triangles(triangleIndex, 2));
      Eigen::Matrix<ScalarT, 3, 2> F = silk::triangleDeformationGradient(x0, x1, x2, invertedRestShape);
      ScalarT potential = silk::baraffWitkinStretchPotential(F);

      double stiffness = stiffnesses(triangleIndex);
      double restArea = triangleRestAreas(triangleIndex);
      double areaFactor = sqrt(restArea);

      ScalarT energy = stiffness * areaFactor * potential;
      return energy;
    });

    std::cout << "n_vars :" << tinyadFunction.n_vars << std::endl;
    return tinyadFunction.eval_with_hessian_proj(x);
  };
  return potentialFunction;
};

int main() {
  VertexPositions vertexPositions;
  vector<Points> pointGroups;
  vector<Edges> edgeGroups;
  vector<Triangles> triangleGroups;
  vector<Tetrahedra> tetrahedraGroups;

  // Add cloth
  auto [clothVertices, clothTriangles] = silk::makeRegularTriangle();

  Eigen::Matrix3d rotationMatrix = Eigen::AngleAxisd(M_PI / 2.0, Eigen::Vector3d::UnitX()).matrix();
  clothVertices = (rotationMatrix * clothVertices.transpose()).transpose();
  clothTriangles = silk::appendTriangles(vertexPositions, triangleGroups, clothVertices, clothTriangles);

  vector<Eigen::Matrix2d> triangleRestShapes = silk::makeRestShapesFromCurrentPositions(vertexPositions,
                                                                                        clothTriangles);

  vector<Eigen::Matrix2d> triangleInvertedRestShapes;
  for (auto &restShape : triangleRestShapes) {
    triangleInvertedRestShapes.push_back(restShape.inverse());
  }

  map<string, PotentialFunction> potentialFunctions;

  int vertexCount = vertexPositions.rows();

  Eigen::VectorXd triangleRestAreas = silk::triangleAreas(vertexPositions, clothTriangles);
  Eigen::VectorXd triangleStretchStiffnesses = Eigen::VectorXd::Ones(clothTriangles.rows());

  potentialFunctions["triangleStretch"] = makePotentialFunction(
      vertexCount, clothTriangles, triangleInvertedRestShapes, triangleStretchStiffnesses, triangleRestAreas);

  vertexPositions(0, 0) += 0.1;
  Eigen::VectorXd x = silk::flatten(vertexPositions);
  auto [f, g, H] = potentialFunctions["triangleStretch"](x);

  std::cout << "f: " << f << std::endl;
  std::cout << "g: " << g << std::endl;

  map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;
  vertexVectorQuantities["stretchForces"] = silk::unflatten(-g);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 5.;
  silk::registerInPolyscope(
      vertexPositions, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups, vertexVectorQuantities);
  polyscope::show();
}