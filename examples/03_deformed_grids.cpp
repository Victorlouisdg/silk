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

int deformation_option = 0;
int grid_resolution = 5;

void update_deformed_mesh() {
  // Create a triangulated grid in 2D
  Eigen::MatrixXd V2;  // Vertex positions in 2D
  Eigen::MatrixXi T;   // Triangles, not we abbreviate as T here to reserve F for the deformeation gradient
  igl::triangulated_grid(grid_resolution, grid_resolution, V2, T);

  // Add a z coordinate to make the triangulated grid 3D
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(V2.rows(), 3);
  V.leftCols(2) = V2;

  switch (deformation_option) {
    case 0:  // Option 0. Simply lifting it 0.5 into the air
      V.col(2) = Eigen::VectorXd::Constant(V.rows(), 0.5);
      break;
    case 1:  // Option 1. Linear deformation z = x + 0.5 * y
      V.col(2) = V.col(0) + 0.5 * V.col(1);
      break;
    case 2:  // Option 2. Quadratic deformation: z = x^2
      V.col(2) = V.col(0).array().square();
      break;
    case 3:  // Option 3. Non-linear deformation: z = x * y + 0.5 * x
      V.col(2) = V.col(0).array() * V.col(1).array() + V.col(0).array();
      break;
    default:
      break;
  }

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

  // Registration in polyscope
  auto *mesh3D = polyscope::registerSurfaceMesh("Triangulated Grid", V, T);
  mesh3D->addFaceScalarQuantity("Stretch Energy", E);
  mesh3D->addFaceScalarQuantity("Area", A);
  mesh3D->setEdgeWidth(1.0);

  auto *mesh2D = polyscope::registerSurfaceMesh2D("Triangulated Grid 2D", V2, T);
  mesh2D->setTransparency(0.5);
  mesh2D->setEdgeWidth(1.0);
}

void callback() {
  bool changed = false;
  changed |= ImGui::InputInt("Grid resolution", &grid_resolution);
  grid_resolution = std::max(grid_resolution, 2);

  changed |= ImGui::RadioButton("Flat", &deformation_option, 0);
  changed |= ImGui::RadioButton("Linear", &deformation_option, 1);
  changed |= ImGui::RadioButton("Quadratic", &deformation_option, 2);
  changed |= ImGui::RadioButton("Non-linear", &deformation_option, 3);

  if (changed) {
    update_deformed_mesh();
  }
}

int main() {
  polyscope::init();
  update_deformed_mesh();
  polyscope::options::automaticallyComputeSceneExtents = false;
  polyscope::state::userCallback = callback;
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::show();
}