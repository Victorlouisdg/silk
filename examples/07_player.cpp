#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <Eigen/Core>
#include <igl/doublearea.h>
#include <igl/triangulated_grid.h>

#include "silk/deformation/deformation_gradient.hh"
#include "silk/deformation/rest_shape.hh"
#include "silk/energy/baraff_witkin.hh"
#include "silk/gui/frame_player.hh"
#include "silk/math/flatten.hh"
#include "silk/math/inverse.hh"
#include "silk/meshes/right_triangle.hh"

#include <TinyAD/ScalarFunction.hh>

int main() {
  // Create a triangulated grid in 2D
  Eigen::MatrixXd V2;  // Vertex positions in 2D
  Eigen::MatrixXi T;   // Triangles, I'm abbreviating as T here to reserve F for the deformation gradient
  igl::triangulated_grid(5, 5, V2, T);

  // Add a z coordinate to make the triangulated grid 3D
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(V2.rows(), 3);
  V.leftCols(2) = V2;

  // Option 3. Quadratic deformation: z = x^2
  V.col(2) = V.col(0).array().square();

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *mesh3D = polyscope::registerSurfaceMesh("Triangulated Grid", V, T);
  mesh3D->setEdgeWidth(1.0);

  int num_frames = 100;

  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::userCallback = [&]() -> void {
    int frame = silk::FramePlayer(num_frames);
    double f = frame / (double) (num_frames - 1);

    Eigen::MatrixXd A = V;
    A.col(2) = V.col(2).array() * f;
    polyscope::registerSurfaceMesh("Triangulated Grid", A, T);
  };
  polyscope::show();
}