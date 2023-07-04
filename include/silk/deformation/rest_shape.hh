#pragma once
#include <Eigen/Core>

namespace silk {

/**
 * @brief Create the rest shape matrix for a triangle using its 2D vertex rest positions. We need this (2 x 2) matrix
 * to calculate the triangle's deformation gradient. See the Dynamic Deformables course notes appendix D formula
 * (D.11). This matrix is sometimes called the "material matrix" and abbreviated D_m.
 *
 * @param u0 The 2D rest position of the first vertex of the triangle.
 * @param u1 The 2D rest position of the second vertex of the triangle.
 * @param u2 The 2D rest position of the third vertex of the triangle.
 * @return The (2 x 2) rest shape matrix.
 */
Eigen::Matrix2d rest_shape(Eigen::Vector2d &u0, Eigen::Vector2d &u1, Eigen::Vector2d &u2) {
  Eigen::Matrix2d Dm;   // The rest shape matrix
  Dm.col(0) = u1 - u0;  // Note that we fill the columns of the matrix, not the rows.
  Dm.col(1) = u2 - u0;
  return Dm;
}

/**
 * @brief Create the rest shape matrix for each triangle using the 2D vertex rest positions. See the docstring of the
 * rest_shape() function for more details.
 *
 * @param V2 (#V x 2) The 2D vertex rest positions.
 * @param F  (#F x 3) The triangles.
 * @return A vector (#F) of (2 x 2) matrices, one for each triangle.
 */
std::vector<Eigen::Matrix2d> rest_shapes(Eigen::MatrixXd &V2, Eigen::MatrixXi &F) {
  std::vector<Eigen::Matrix2d> Dm;
  Dm.reserve(F.rows());

  for (int i = 0; i < F.rows(); i++) {
    Eigen::Vector2d u0 = V2.row(F(i, 0));
    Eigen::Vector2d u1 = V2.row(F(i, 1));
    Eigen::Vector2d u2 = V2.row(F(i, 2));
    Eigen::Matrix2d Dm_i = rest_shape(u0, u1, u2);
    Dm.push_back(Dm_i);
  }

  return Dm;
}
}  // namespace silk