#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <iostream>

#include <TinyAD/ScalarFunction.hh>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace silk {

std::vector<Eigen::Matrix3d> initializeTetrahedronRestShapes(Eigen::MatrixXd &V, Eigen::MatrixXi &T) {
  std::vector<Eigen::Matrix3d> restShapes;
  restShapes.reserve(T.rows());

  // This notation is from the TinyAD examples.
  for (int t_idx = 0; t_idx < T.rows(); ++t_idx) {
    // Get 3D vertex positions
    Eigen::Vector3d ar = V.row(T(t_idx, 0));
    Eigen::Vector3d br = V.row(T(t_idx, 1));
    Eigen::Vector3d cr = V.row(T(t_idx, 2));
    Eigen::Vector3d dr = V.row(T(t_idx, 3));

    // Save 3-by-3 matrix with edge vectors as colums
    restShapes[t_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
  };
  return restShapes;
}

template<typename T3, typename T2>
Eigen::Matrix<T3, 3, 2> deformationGradient(Eigen::Vector3<T3> vertexPosition0,
                                            Eigen::Vector3<T3> vertexPosition1,
                                            Eigen::Vector3<T3> vertexPosition2,
                                            Eigen::Vector2<T2> vertexRestPosition0,
                                            Eigen::Vector2<T2> vertexRestPosition1,
                                            Eigen::Vector2<T2> vertexRestPosition2) {
  Eigen::Vector3<T3> x0 = vertexPosition0;
  Eigen::Vector3<T3> x1 = vertexPosition1;
  Eigen::Vector3<T3> x2 = vertexPosition2;

  Eigen::Vector2<T2> u0 = vertexRestPosition0;
  Eigen::Vector2<T2> u1 = vertexRestPosition1;
  Eigen::Vector2<T2> u2 = vertexRestPosition2;

  // TODO: if the triangle rest shape is static, this matrix (and its inverse) can be precomputed.
  // This could be supported via overloaded versions of the deformationGradient function.
  Eigen::Matrix2d Dm;
  Dm.col(0) = u1 - u0;
  Dm.col(1) = u2 - u0;
  Eigen::Matrix2d Dm_inv = Dm.inverse();

  Eigen::Matrix<T3, 3, 2> Ds;
  Ds.col(0) = x1 - x0;
  Ds.col(1) = x2 - x0;

  Eigen::Matrix<T3, 3, 2> F = Ds * Dm_inv;
  return F;
};

template<typename T> T baraffWitkinStretchPotential(Eigen::Matrix<T, 3, 2> deformationGradient) {
  Eigen::Matrix<T, 3, 2> F = deformationGradient;
  Eigen::Matrix<T, 3, 1> wu = F.col(0);
  Eigen::Matrix<T, 3, 1> wv = F.col(1);

  T Cu = wu.norm() - 1.0;
  T Cv = wv.norm() - 1.0;

  T Eu = 0.5 * Cu * Cu;
  T Ev = 0.5 * Cv * Cv;
  T E = Eu + Ev;
  return E;
}

template<typename T> T baraffWitkinShearPotential(Eigen::Matrix<T, 3, 2> deformationGradient) {
  Eigen::Matrix<T, 3, 2> F = deformationGradient;
  Eigen::Matrix<T, 3, 1> wu = F.col(0);
  Eigen::Matrix<T, 3, 1> wv = F.col(1);

  // T C = wu.dot(wv);
  T C = wu.transpose() * wv;
  T E = 0.5 * C * C;
  return E;
}

}  // namespace silk