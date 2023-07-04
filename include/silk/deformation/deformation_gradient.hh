#pragma once
#include <Eigen/Core>

#include "silk/deformation/deformed_shape.hh"
#include "silk/deformation/rest_shape.hh"
#include "silk/math/inverse.hh"

namespace silk {

/**
 * @brief Calculate the deformation gradient for a triangle using its 3D vertex positions and the inverse of its rest
 * shape matrix. The deformation gradient is measure of how much the triangle has deformed from its rest shape. To
 * understand this concept, read Chapter 2 of the Dynamic Deformables course notes or watch the accompanying video.
 *
 * @tparam AD Type that may or may not be an autodiff type.
 * @tparam T Type that is compatible with the AD type, but usually isn't an autodiff type.
 * @param x0 The 3D position of the first vertex of the triangle.
 * @param x1 The 3D position of the second vertex of the triangle.
 * @param x2 The 3D position of the third vertex of the triangle.
 * @param Dm_inv The inverse of the rest shape matrix of the triangle.
 * @return The (3 x 2) deformation gradient F of the triangle.
 */
template<typename AD, typename T>
Eigen::Matrix<AD, 3, 2> deformation_gradient(Eigen::Vector3<AD> x0,
                                             Eigen::Vector3<AD> x1,
                                             Eigen::Vector3<AD> x2,
                                             Eigen::Matrix<T, 2, 2> Dm_inv) {
  Eigen::Matrix<AD, 3, 2> Ds = deformed_shape(x0, x1, x2);
  Eigen::Matrix<AD, 3, 2> F = Ds * Dm_inv;  // Formula (D.11) in the Dynamic Deformables course notes.
  return F;
};

/**
 * @brief Calculate the deformation gradient of the triangles, using the inverse of their rest shape matrices. Use
 *
 * @param V (#V x 3) matrix of vertex positions.
 * @param T (#T x 3) matrix of triangle indices.
 * @param Dm_inv A vector (#T) with the inverse of the rest shape matrix of each triangle.
 * @return A vector (#T) of with (3 x 2) deformation gradients for each triangle.
 */
std::vector<Eigen::Matrix<double, 3, 2>> deformation_gradients(Eigen::MatrixXd &V,
                                                               Eigen::MatrixXi &T,
                                                               std::vector<Eigen::Matrix2d> &Dm_inv) {
  std::vector<Eigen::Matrix<double, 3, 2>> F;
  F.reserve(T.rows());

  for (int i = 0; i < T.rows(); i++) {
    Eigen::Vector3d x0 = V.row(T(i, 0));
    Eigen::Vector3d x1 = V.row(T(i, 1));
    Eigen::Vector3d x2 = V.row(T(i, 2));
    Eigen::Matrix<double, 3, 2> Fi = silk::deformation_gradient(x0, x1, x2, Dm_inv[i]);
    F.push_back(Fi);
  }
  return F;
}

/**
 * @brief Calculate the deformation gradient of the triangles, using the 2D rest positions of the vertices.
 *
 * @param V (#V x 3) matrix of vertex positions.
 * @param T (#T x 3) matrix of triangle indices.
 * @param V2 (#V x 2) matrix of vertex rest positions.
 * @return A vector (#T) of with (3 x 2) deformation gradients for each triangle.
 */
std::vector<Eigen::Matrix<double, 3, 2>> deformation_gradients(Eigen::MatrixXd &V,
                                                               Eigen::MatrixXi &T,
                                                               Eigen::MatrixXd &V2) {
  // Created the rest shape matrices and inverted them
  std::vector<Eigen::Matrix2d> Dm = silk::rest_shapes(V2, T);
  std::vector<Eigen::Matrix2d> Dm_inv = silk::inverse(Dm);

  // Use those to calculate the deformation gradients
  std::vector<Eigen::Matrix<double, 3, 2>> F = silk::deformation_gradients(V, T, Dm_inv);
  return F;
}

}  // namespace silk