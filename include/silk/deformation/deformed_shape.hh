#pragma once
#include <Eigen/Core>

namespace silk {

/**
 * @brief Create the deformed shape matrix for a triangle using its 3D vertex positions. We need this (3 x 2) matrix
 * to calculate the triangle's deformation gradient. See the Dynamic Deformables course notes appendix D formula
 * (D.11). This matrix is sometimes abbreviated D_s where the s stands for "spatial coordinates".
 *
 * @tparam T The datatype of the positions. Templated so that we can use it with TinyAD's autodiff types.
 * @param x0 The 3D position of the first vertex of the triangle.
 * @param x1 The 3D position of the second vertex of the triangle.
 * @param x2 The 3D position of the third vertex of the triangle.
 * @return The (3 x 2) matrix that captures the deformed shape of the triangle.
 */
template<typename T>
Eigen::Matrix<T, 3, 2> deformed_shape(Eigen::Vector3<T> &x0, Eigen::Vector3<T> &x1, Eigen::Vector3<T> &x2) {
  Eigen::Matrix<T, 3, 2> ds;
  ds.col(0) = x1 - x0;
  ds.col(1) = x2 - x0;
  return ds;
}
}  // namespace silk