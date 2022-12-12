#include <TinyAD/Support/GeometryCentral.hh>

#include <TinyAD/Scalar.hh>
#include <TinyAD/ScalarFunction.hh>

#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "silk/conversions.hh"

#include "silk/energies.hh"
#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <gtest/gtest.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

using namespace silk;

// Demonstrate some basic assertions.
TEST(DeformationGradient, BasicAssertions) {
  unique_ptr<ManifoldSurfaceMesh> mesh_pointer;
  unique_ptr<VertexPositionGeometry> geometry_pointer;
  std::tie(mesh_pointer, geometry_pointer) = silk::makeSingleTriangle();
  ManifoldSurfaceMesh &mesh = *mesh_pointer;
  VertexPositionGeometry &geometry = *geometry_pointer;

  geometry.requireVertexPositions();
  VertexData<Eigen::Vector2d> vertexRestPositions = silk::makeProjectedRestPositions(mesh, geometry);

  Face f = *(mesh.faces().begin());
  Vertex v0 = f.halfedge().vertex();
  Vertex v1 = f.halfedge().next().vertex();
  Vertex v2 = f.halfedge().next().next().vertex();
  Eigen::Vector3d x0 = to_eigen(geometry.vertexPositions[v0]);
  Eigen::Vector3d x1 = to_eigen(geometry.vertexPositions[v1]);
  Eigen::Vector3d x2 = to_eigen(geometry.vertexPositions[v2]);

  Eigen::Vector2d u0 = vertexRestPositions[v0];
  Eigen::Vector2d u1 = vertexRestPositions[v1];
  Eigen::Vector2d u2 = vertexRestPositions[v2];

  // Deformation gradient should be permutation invariant.
  Eigen::Matrix<double, 3, 2> F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
  Eigen::Matrix<double, 3, 2> F2 = silk::deformationGradient(x1, x2, x0, u1, u2, u0);
  Eigen::Matrix<double, 3, 2> F3 = silk::deformationGradient(x2, x0, x1, u2, u0, u1);
  Eigen::Matrix<double, 3, 2> F4 = silk::deformationGradient(x0, x2, x1, u0, u2, u1);
  Eigen::Matrix<double, 3, 2> F5 = silk::deformationGradient(x1, x0, x2, u1, u0, u2);
  Eigen::Matrix<double, 3, 2> F6 = silk::deformationGradient(x2, x1, x0, u2, u1, u0);
  EXPECT_TRUE(F.isApprox(F2));
  EXPECT_TRUE(F.isApprox(F3));
  EXPECT_TRUE(F.isApprox(F4));
  EXPECT_TRUE(F.isApprox(F5));
  EXPECT_TRUE(F.isApprox(F6));

  // Note: these equalities only hold when there is no translation, because:
  // x = affine_transform(u) = F * u + t
  EXPECT_TRUE(x0.isApprox(F * u0));
  EXPECT_TRUE(x1.isApprox(F * u1));
  EXPECT_TRUE(x2.isApprox(F * u2));

  Eigen::Matrix<double, 3, 2> F_expected;
  F_expected.col(0) << 1, 0, 0;
  F_expected.col(1) << 0, 1, 0;

  EXPECT_TRUE(F.isApprox(F_expected));

  Eigen::Vector3d t(0.1, 0.2, 0.3);
  x0 += t;
  x1 += t;
  x2 += t;
  F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
  EXPECT_TRUE(F.isApprox(F_expected)) << "The deformation gradient should be invariant to world space translation.";

  Eigen::Vector2d t2(0.4, 0.5);
  u0 += t2;
  u1 += t2;
  u2 += t2;
  F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
  EXPECT_TRUE(F.isApprox(F_expected)) << "The deformation gradient should be invariant to material space translation.";
  //   EXPECT_DOUBLE_EQ(F, F_expected)

  // Reset positions
  x0 = to_eigen(geometry.vertexPositions[v0]);
  x1 = to_eigen(geometry.vertexPositions[v1]);
  x2 = to_eigen(geometry.vertexPositions[v2]);
  u0 = vertexRestPositions[v0];
  u1 = vertexRestPositions[v1];
  u2 = vertexRestPositions[v2];
  x0 *= 2.0;
  x1 *= 2.0;
  x2 *= 2.0;
  F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
  EXPECT_TRUE(F.isApprox(2.0 * F_expected)) << "The deformation gradient should scale linearly.";
}