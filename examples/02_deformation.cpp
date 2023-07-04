#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <Eigen/Core>
#include <igl/doublearea.h>
#include <igl/triangulated_grid.h>

#include "silk/deformation/deformation_gradient.hh"
#include "silk/deformation/rest_shape.hh"
#include "silk/math/inverse.hh"
#include "silk/meshes/right_triangle.hh"

int main() {
  /**
   * @brief
   */

  // Create a triangulated grid in 2D
  Eigen::MatrixXd V2;  // Vertex positions in 2D
  Eigen::MatrixXi T;   // Triangles, not we abbreviate as T here to reserve F for the deformeation gradient
  igl::triangulated_grid(5, 5, V2, T);

  // Add a z coordinate to make the triangulated grid 3D
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(V2.rows(), 3);
  V.leftCols(2) = V2;
  V.col(2) = Eigen::VectorXd::Constant(V.rows(), 0.5);

  // Compute the deformation gradients
  std::vector<Eigen::Matrix<double, 3, 2>> F = silk::deformation_gradients(V, T, V2);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::registerSurfaceMesh("Triangulated Grid", V, T);
  auto *psMesh2D = polyscope::registerSurfaceMesh2D("Triangulated Grid 2D", V2, T);
  psMesh2D->setTransparency(0.5);
  polyscope::show();
}