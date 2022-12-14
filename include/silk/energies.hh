#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <iostream>

#include <TinyAD/ScalarFunction.hh>

#include "silk/types.hh"

using namespace std;

namespace silk {

/**
 * @brief Compute the inverse of the rest shape matrix for each triangle. The rest shape matrix is a 2x2 matrix that
 * contains are column the relative vector going from v0 to v1 an v0 to v2. It is also sometimes called the material
 * matrix. The inverse of this matrix is needed to calculate the triangle's deformation gradient.
 *
 * @param vertexPositions
 * @param triangles
 * @return for each triangle its inverted rest shape matrix.
 */
vector<Eigen::Matrix2d> initializeInvertedTriangleRestShapes(const VertexPositions &vertexPositions,
                                                             const Triangles &triangles) {

  vector<Eigen::Matrix2d> invertedRestShapes;
  invertedRestShapes.reserve(triangles.rows());

  for (int triangleIndex = 0; triangleIndex < triangles.rows(); ++triangleIndex) {
    Eigen::Vector3d x0 = vertexPositions.row(triangles(triangleIndex, 0));
    Eigen::Vector3d x1 = vertexPositions.row(triangles(triangleIndex, 1));
    Eigen::Vector3d x2 = vertexPositions.row(triangles(triangleIndex, 2));

    // Create surrogates for u1 - u0 and u2 - u0, x1x0 is aligned with the u-axis in 2D by convention.
    // Other choices are possible and valid. This type of initialization is not well suited for anisotropic materials.
    // Because the choice of anisotropy directions is quite abritrary.
    Eigen::Vector3d x1x0 = x1 - x0;
    Eigen::Vector3d x2x0 = x2 - x0;

    Eigen::Vector2d u1u0(x1x0.norm(), 0.0);

    // Angle between (x1 -x0) and (x2 - x0)
    double angle = acos(x1x0.dot(x2x0) / (x1x0.norm() * (x2x0).norm()));
    Eigen::Vector2d u2u0(cos(angle) * (x2x0).norm(), sin(angle) * (x2x0).norm());

    Eigen::Matrix2d restShape;
    restShape << u1u0, u2u0;  // It is important that these are column vectors.
    invertedRestShapes[triangleIndex] = restShape.inverse();
  };
  return invertedRestShapes;
}

vector<Eigen::Matrix3d> initializeInvertedTetrahedronRestShapes(VertexPositions const &vertexPositions,
                                                                Tetrahedra const &tetrahedra) {

  vector<Eigen::Matrix3d> invertedRestShapes;
  invertedRestShapes.reserve(tetrahedra.rows());

  for (int tetrahedronIndex = 0; tetrahedronIndex < tetrahedra.rows(); ++tetrahedronIndex) {
    Eigen::Vector3d x0 = vertexPositions.row(tetrahedra(tetrahedronIndex, 0));
    Eigen::Vector3d x1 = vertexPositions.row(tetrahedra(tetrahedronIndex, 1));
    Eigen::Vector3d x2 = vertexPositions.row(tetrahedra(tetrahedronIndex, 2));
    Eigen::Vector3d x3 = vertexPositions.row(tetrahedra(tetrahedronIndex, 3));

    Eigen::Matrix3d restShape;
    restShape << x1 - x0, x2 - x0, x3 - x0;  // It is important that these are column vectors.
    invertedRestShapes[tetrahedronIndex] = restShape.inverse();
  };
  return invertedRestShapes;
}

std::vector<Eigen::Matrix3d> initializeTetrahedronRestShapes(Eigen::MatrixXd const &V, Eigen::MatrixXi const &T) {
  std::vector<Eigen::Matrix3d> restShapes;
  restShapes.reserve(T.rows());

  // This notation is from the TinyAD examples.
  for (int t_idx = 0; t_idx < T.rows(); ++t_idx) {
    // Get 3D vertex positions
    Eigen::Vector3d x0 = V.row(T(t_idx, 0));
    Eigen::Vector3d x1 = V.row(T(t_idx, 1));
    Eigen::Vector3d x2 = V.row(T(t_idx, 2));
    Eigen::Vector3d x3 = V.row(T(t_idx, 3));

    // Save 3-by-3 matrix with edge vectors as colums
    restShapes[t_idx] = TinyAD::col_mat(x1 - x0, x2 - x0, x3 - x0);
  };
  return restShapes;
}

template<typename T3, typename T2>
Eigen::Matrix<T3, 3, 2> triangleDeformationGradient(Eigen::Vector3<T3> vertexPosition0,
                                                    Eigen::Vector3<T3> vertexPosition1,
                                                    Eigen::Vector3<T3> vertexPosition2,
                                                    Eigen::Matrix<T2, 2, 2> invertedRestShape) {
  // Notation from the Dynamics Deformables course notes.
  Eigen::Vector3<T3> x0 = vertexPosition0;
  Eigen::Vector3<T3> x1 = vertexPosition1;
  Eigen::Vector3<T3> x2 = vertexPosition2;

  Eigen::Matrix<T3, 3, 2> Ds;
  Ds.col(0) = x1 - x0;
  Ds.col(1) = x2 - x0;

  Eigen::Matrix2d Dm_inv = invertedRestShape;

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