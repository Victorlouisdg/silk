#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "silk/conversions.hh"
#include "silk/deformation/rest_shapes.hh"
#include "silk/energies.hh"
#include "silk/energy/energy_mapping.hh"
#include "silk/geometry/area.hh"
#include "silk/mesh_construction.hh"
#include "silk/simple_meshes.hh"
#include "silk/types.hh"
#include "silk/visualization.hh"

#include <igl/triangle/triangulate.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

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

  map<string, EnergyFunction> energyFunctions;

  int vertexCount = vertexPositions.rows();

  Eigen::VectorXd triangleRestAreas = silk::triangleAreas(vertexPositions, clothTriangles);
  Eigen::VectorXd triangleStretchStiffnesses = Eigen::VectorXd::Ones(clothTriangles.rows());

  // Adding the stretch energy
  TinyAD::ScalarFunction<3, double, Eigen::Index> triangleStretchScalarFunction = silk::createTriangleScalarFunction(
      [](auto &&F) { return silk::baraffWitkinStretchPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);
  EnergyFunction triangleStretchFunction = silk::standardizeScalarFunction(triangleStretchScalarFunction);
  energyFunctions["triangleStretch"] = triangleStretchFunction;

  // Adding the shear energy
  TinyAD::ScalarFunction<3, double, Eigen::Index> triangleShearScalarFunction = silk::createTriangleScalarFunction(
      [](auto &&F) { return silk::baraffWitkinShearPotential(F); },
      vertexCount,
      clothTriangles,
      triangleInvertedRestShapes);
  EnergyFunction triangleShearFunction = silk::standardizeScalarFunction(triangleShearScalarFunction);
  energyFunctions["triangleShear"] = triangleShearFunction;

  vertexPositions(0, 0) += 0.1;
  Eigen::VectorXd x = silk::flatten(vertexPositions);

  map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexVectorQuantities;
  for (auto const &[name, energyFunction] : energyFunctions) {
    auto [f, g, H] = energyFunction(x);
    cout << name << ": " << f << endl;
    vertexVectorQuantities[name] = silk::unflatten(-g);
  }

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 5.;
  silk::registerInPolyscope(
      vertexPositions, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups, vertexVectorQuantities);
  polyscope::show();
}