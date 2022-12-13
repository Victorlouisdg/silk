#include "silk/visualization.hh"
#include <iostream>

using namespace std;

int main() {
  Eigen::Matrix<double, Eigen::Dynamic, 3> vertexPositions;

  // Add the spring
  Eigen::RowVector3d v0(0.0, 0.0, 3.0);
  Eigen::RowVector3d v1(0.0, 0.0, 2.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v0, v1;
  Eigen::ArrayX2i springEdge(1, 2);
  springEdge << 0, 1;

  // Add the triangle
  Eigen::RowVector3d v2(1.0, 0.0, 1.0);
  Eigen::RowVector3d v3(-1.0, 0.0, 1.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v2, v3;
  Eigen::ArrayX3i triangle(1, 3);
  triangle << 1, 2, 3;

  // Add the tetrahedron
  Eigen::RowVector3d v4(0.0, 1.0, 0.0);
  Eigen::RowVector3d v5(0.0, -1.0, 0.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v4, v5;
  Eigen::ArrayX4i tetrahedron(1, 4);
  tetrahedron << 2, 3, 4, 5;

  // Add an extra point
  Eigen::RowVector3d v6(0.0, 0.0, 4.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 1, 3);
  vertexPositions.bottomRows(1) << v6;
  Eigen::ArrayXi points_(3);
  points_ << 0, 1, 6;

  vector<Eigen::ArrayXi> points{points_};
  vector<Eigen::ArrayX2i> edges{springEdge};
  vector<Eigen::ArrayX3i> triangles{triangle};
  vector<Eigen::ArrayX4i> tetrahedra{tetrahedron};

  std::cout << vertexPositions << std::endl;

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  silk::registerInPolyscope(vertexPositions, points, edges, triangles, tetrahedra);
  polyscope::show();
}