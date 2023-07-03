#include "polyscope/standardize_data_array.h"
#include "silk/types.hh"
#include <igl/triangle/triangulate.h>
#include <iostream>
#include <silk/simple_meshes.hh>

namespace silk {
using namespace Eigen;

Eigen::Vector3d make3D(Eigen::Vector2d v2) {
  Eigen::Vector3d v3;
  v3 << v2, 0.0;
  return v3;
}

tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeRegularTetrahedron() {
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(4, 3);  // 3D vertex positions

  // Adapted from:  https://www.danielsieger.com/blog/2021/01/03/generating-platonic-solids.html
  // choose coordinates on the unit sphere
  double a = 1.0 / 3.0;
  double b = sqrt(8.0 / 9.0);
  double c = sqrt(2.0 / 9.0);
  double d = sqrt(2.0 / 3.0);

  Eigen::RowVector3d v0(0, 0, 1);
  Eigen::RowVector3d v2(-c, d, -a);
  Eigen::RowVector3d v1(-c, -d, -a);
  Eigen::RowVector3d v3(b, 0, -a);
  V << v0, v1, v2, v3;

  Eigen::MatrixXi T(1, 4);  // tetrahedra
  T << 0, 1, 2, 3;

  return std::make_tuple(V, T);
};

tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeStackedTetrahedra() {
  Eigen::MatrixXd V0;
  Eigen::MatrixXi T0;
  // Eigen::Matrix<int, Eigen::Dynamic, 4> T0;  // Test whether this results in efficiency gains
  std::tie(V0, T0) = silk::makeRegularTetrahedron();

  Eigen::MatrixXd V1;
  Eigen::MatrixXi T1;
  std::tie(V1, T1) = silk::makeRegularTetrahedron();
  V1.rightCols(1).array() += 1.5;  // Move tetrahedron up

  // Combine arrays of the tets for joint processing.
  Eigen::MatrixXd V = Eigen::MatrixXd(V0.rows() + V1.rows(), V0.cols());
  V.topRows(V0.rows()) = V0;
  V.bottomRows(V1.rows()) = V1;

  Eigen::MatrixXi T = Eigen::MatrixXi(T0.rows() + T1.rows(), T0.cols());
  T.topRows(T0.rows()) = T0;
  // Account for the offset the vertices of the second tetrahedron by the amount of vertices in the first.
  T.bottomRows(T1.rows()) = T1.array() + V0.rows();

  return std::make_tuple(V, T);
};

tuple<VertexPositions, Triangles> makeBox() {
  Eigen::RowVector3d v0(-1.0, -1.0, 0.0);
  Eigen::RowVector3d v1(1.0, -1.0, 0.0);
  Eigen::RowVector3d v2(1.0, 1.0, 0.0);
  Eigen::RowVector3d v3(-1.0, 1.0, 0.0);

  Eigen::RowVector3d boxHeight(0.0, 0.0, 2.0);
  Eigen::RowVector3d v4 = v0 + boxHeight;
  Eigen::RowVector3d v5 = v1 + boxHeight;
  Eigen::RowVector3d v6 = v2 + boxHeight;
  Eigen::RowVector3d v7 = v3 + boxHeight;

  Eigen::Matrix<double, 8, 3> vertexCoordinates(8, 3);
  vertexCoordinates << v0, v1, v2, v3, v4, v5, v6, v7;

  // Perspective: look from (-1,-1, -1) to the origin.
  // Vertices are ordered such that normals point inwards.

  // Base
  Eigen::RowVector3i triangle0{0, 1, 2};
  Eigen::RowVector3i triangle1{0, 2, 3};

  // Front
  Eigen::RowVector3i triangle2{0, 1, 5};
  Eigen::RowVector3i triangle3{0, 5, 4};

  // Right
  Eigen::RowVector3i triangle4{1, 6, 2};
  Eigen::RowVector3i triangle5{1, 5, 6};

  // Back
  Eigen::RowVector3i triangle6{3, 2, 6};
  Eigen::RowVector3i triangle7{3, 6, 7};

  // Left
  Eigen::RowVector3i triangle8{0, 3, 7};
  Eigen::RowVector3i triangle9{0, 7, 4};

  Triangles triangles(10, 3);
  triangles << triangle0, triangle1, triangle2, triangle3, triangle4, triangle5, triangle6, triangle7, triangle8,
      triangle9;

  return std::make_tuple(vertexCoordinates, triangles);
}

tuple<VertexPositions, Triangles> makeTriangulatedSquare() {
  RowVector3d v0(-1.0, -1.0, 0.0);
  RowVector3d v1(1.0, -1.0, 0.0);
  RowVector3d v2(1.0, 1.0, 0.0);
  RowVector3d v3(-1.0, 1.0, 0.0);

  Matrix<double, 4, 3> vertexCoordinates(4, 3);
  vertexCoordinates << v0, v1, v2, v3;

  MatrixXi edges(4, 2);  // Mesh edges
  edges << 0, 1, 1, 2, 2, 3, 3, 0;

  MatrixXd vertexCoordinates2D = vertexCoordinates.leftCols(2);

  MatrixXd newVertices2D;
  MatrixXi newTriangles;

  MatrixXd H;

  igl::triangle::triangulate(vertexCoordinates2D, edges, H, "a0.001q", newVertices2D, newTriangles);

  MatrixXd newVertexCoordinates = MatrixXd::Zero(newVertices2D.rows(), 3);
  newVertexCoordinates.leftCols(2) = newVertices2D;

  return std::make_tuple(newVertexCoordinates, newTriangles);
}

}  // namespace silk
