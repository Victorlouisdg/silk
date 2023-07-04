#pragma once

#include <Eigen/Core>

/**
 * @brief In this file we implement the energy functions from Baraff and Witkin's paper "Large Steps in Cloth 
 * Simulation (1998)". However, we mostly prefer the perspective and notation form Theodore Kim's "A Finite Element 
 * Formulation of Baraff-Witkin Cloth (2020)".
 */

namespace silk {

/**
 * @brief Calculate the stretch energy of a triangle using its deformation gradient and rest area. The bu and bv 
 * parameters are described just below equation (10) in Baraff and Witkin's paper. Note that u and v are directions in
 * the space of the triangle's 2D rest positions.
 *
 * @tparam T The type of the deformation gradient, templated for autodiff.
 * @param F The (3 x 2) deformation gradient of the triangle.
 * @param a The rest area of the triangle.
 * @param bu Default value is 1.0, can be used to control shrink/stretch in the u direction.
 * @param bv Default value is 1.0, can be used to control shrink/stretch in the v direction.
 */
template<typename T> T stretch_bw(Eigen::Matrix<T, 3, 2> F, double a, double bu = 1.0, double bv = 1.0) {
  Eigen::Matrix<T, 3, 1> wu = F.col(0);
  Eigen::Matrix<T, 3, 1> wv = F.col(1);

  // These abbreviations with "C" are inspired by Baraff Witkins notation e.g. in equation (10). 
  T Cu = wu.norm() - bu;
  T Cv = wv.norm() - bv;

  T Eu = Cu * Cu;
  T Ev = Cv * Cv;
  T E = a * (Eu + Ev);
  return E;
}

}  // namespace silk