#include "polyscope/standardize_data_array.h"
#include <igl/triangle/triangulate.h>
#include <iostream>
#include <silk/simple_meshes.hh>

namespace silk {

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

}  // namespace silk
