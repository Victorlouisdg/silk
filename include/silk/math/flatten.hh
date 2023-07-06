#pragma once

#include <Eigen/Core>

namespace silk {

/**
 * @brief Flatten a (N x 3) matrix into a (3N x 1) vector by stacking the rows. I believe this function does the same
 * thing as the TinyAD func.x_from_data function (i.e. row-major vectorization).
 * Also see this note: https://github.com/alecjacobson/libigl-tinyad-example/tree/main#ordering
 *
 * TODO clarify what happens to the original matrix.
 *
 * @param M The (N x 3) matrix to flatten.
 * @return The vector obtained by flattening the matrix.
 */
Eigen::VectorXd flatten(Eigen::MatrixXd const &M) {
  return M.reshaped<Eigen::RowMajor>(M.rows() * 3, 1);
}

/**
 * @brief Unflatten a (3N x 1) vector into a (N x 3) matrix by unstacking the rows.
 * 
 * @param v The flat (3N x 1) vector to unflatten.
 * @return The (N x 3) matrix obtained by unflattening the vector.
 */
Eigen::MatrixXd unflatten(Eigen::VectorXd const &v) {
  // TODO: check if vector has a multiple of 3 rows
  assert (v.rows() % 3 == 0);
  return v.reshaped<Eigen::RowMajor>(v.rows() / 3, 3);
}

}  // namespace silk