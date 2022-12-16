#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <igl/triangle/triangulate.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

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

  igl::triangle::triangulate(vertexCoordinates2D, edges, H, "a0.1q", newVertices2D, newTriangles);

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

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::lengthScale = 5.;

  silk::registerInPolyscope(vertexPositions, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups);

  polyscope::show();
}