#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "silk/meshes/right_triangle.hh"
#include <Eigen/Core>
#include <igl/doublearea.h>

int main() {
  /**
   * @brief Basic example combining polyscope, igl and silk.
   */
  Eigen::MatrixXd V;
  Eigen::MatrixXi F;
  silk::right_triangle(V, F);

  // Compute the area of the triangle with libigl
  Eigen::VectorXd areas;
  igl::doublearea(V, F, areas);
  areas = 0.5 * areas.array();

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::registerSurfaceMesh("Right triangle", V, F);
  polyscope::getSurfaceMesh("Right triangle")->addFaceScalarQuantity("area", areas);
  polyscope::show();
}