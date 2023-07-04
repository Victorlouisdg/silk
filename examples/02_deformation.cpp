#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <Eigen/Core>
#include <igl/doublearea.h>
#include <igl/triangulated_grid.h>

#include "silk/deformation/deformation_gradient.hh"
#include "silk/deformation/rest_shape.hh"
#include "silk/energy/baraff_witkin.hh"
#include "silk/math/inverse.hh"
#include "silk/meshes/right_triangle.hh"

int main() {
  // Create a triangulated grid in 2D
  Eigen::MatrixXd V2;  // Vertex positions in 2D
  Eigen::MatrixXi T;   // Triangles, not we abbreviate as T here to reserve F for the deformeation gradient
  igl::triangulated_grid(5, 5, V2, T);

  // Add a z coordinate to make the triangulated grid 3D
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(V2.rows(), 3);
  V.leftCols(2) = V2;

  // Option 3. Quadratic deformation: z = x^2 
  V.col(2) = V.col(0).array().square();

  // Compute the deformation gradients
  std::vector<Eigen::Matrix<double, 3, 2>> F = silk::deformation_gradients(V, T, V2);
  Eigen::VectorXd A2;  // area in 2D of rest shape triangles
  igl::doublearea(V2, T, A2);
  A2 = 0.5 * A2.array();

  // Compute an elastic potential energy for each triangle
  std::vector<double> E;
  E.reserve(T.rows());
  for (int i = 0; i < T.rows(); i++) {
    double Ei = silk::stretch_bw(F[i], A2[i]);
    E.push_back(Ei);
  }

  // Compute the areas in 3D of the deformed triangles
  Eigen::VectorXd A;
  igl::doublearea(V, T, A);
  A = 0.5 * A.array();

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *mesh3D = polyscope::registerSurfaceMesh("Triangulated Grid", V, T);
  mesh3D->addFaceScalarQuantity("E", E);
  mesh3D->addFaceScalarQuantity("A", A);
  mesh3D->setEdgeWidth(1.0);

  auto *mesh2D = polyscope::registerSurfaceMesh2D("Triangulated Grid 2D", V2, T);
  mesh2D->setTransparency(0.5);
  mesh2D->setEdgeWidth(1.0);
  polyscope::show();
}