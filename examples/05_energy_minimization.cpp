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

void setup_scene(Eigen::MatrixXd &V,
                 Eigen::MatrixXd &V2,
                 Eigen::MatrixXi &T,
                 Eigen::VectorXd &A2,
                 std::vector<Eigen::Matrix2d> &Dm_inv) {
  // Create a triangulated grid in 2D
  igl::triangulated_grid(5, 5, V2, T);

  // Add a z coordinate to make the triangulated grid 3D
  V = Eigen::MatrixXd::Zero(V2.rows(), 3);
  V.leftCols(2) = V2;

  // Option 3. Quadratic deformation: z = x^2
  V.col(2) = V.col(0).array().square();

  // Compute the rest areas
  igl::doublearea(V2, T, A2);
  A2 = 0.5 * A2.array();

  // Created the rest shape matrices and inverted them
  std::vector<Eigen::Matrix2d> Dm = silk::rest_shapes(V2, T);
  Dm_inv = silk::inverse(Dm);
}

int main() {
  Eigen::MatrixXd V;   // Vertex positions in 3D
  Eigen::MatrixXd V2;  // Vertex positions in 2D
  Eigen::MatrixXi T;   // Triangles, not we abbreviate as T here to reserve F for the deformeation gradient
  Eigen::VectorXd A2;  // Triangle areas in 2D
  std::vector<Eigen::Matrix2d> Dm_inv;  // Inverse rest shape matrices
  setup_scene(V, V2, T, A2, Dm_inv);
  double k = 1.0;  // Stretch stiffness, uniform for all triangles

  // TinyAD usage example for per-triangle energy using the deformation gradient
  // This is a bit more verbose than the simple for each loop in 02_deformation.cpp, however TinyAD can now provide us
  // with the gradient of the energy with respect to the vertex positions, which we need for the simulation.
  auto func = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));
  func.add_elements<3>(TinyAD::range(T.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    using ST = TINYAD_SCALAR_TYPE(element);
    Eigen::Index i = element.handle;  // triangle index
    Eigen::Vector3<ST> x0 = element.variables(T(i, 0));
    Eigen::Vector3<ST> x1 = element.variables(T(i, 1));
    Eigen::Vector3<ST> x2 = element.variables(T(i, 2));

    Eigen::Matrix<ST, 3, 2> F = silk::deformation_gradient(x0, x1, x2, Dm_inv[i]);
    double a = A2(i);                   // area of the rest shape triangle
    ST E = k * silk::stretch_bw(F, a);  // stretch energy of the triangle
    return E;
  });

  std::vector<Eigen::MatrixXd> V_hist;  // history of vertex positions
  V_hist.push_back(V);

  // Gradient descent
  int num_iterations = 100;
  for (int i = 0; i < num_iterations; i++) {
    Eigen::VectorXd x = silk::flatten(V);
    auto [f, g, H_proj] = func.eval_with_hessian_proj(x);
    Eigen::MatrixXd forces = -silk::unflatten(g);
    V += 0.1 * forces;
    V_hist.push_back(V);
  }

  // PolyScope configuration
  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  polyscope::options::automaticallyComputeSceneExtents = false;

  // First registration of the meshes, to set some options
  auto *mesh3D = polyscope::registerSurfaceMesh("Triangulated Grid", V, T);
  mesh3D->setEdgeWidth(1.0);

  auto *mesh2D = polyscope::registerSurfaceMesh2D("Triangulated Grid 2D", V2, T);
  mesh2D->setTransparency(0.5);
  mesh2D->setEdgeWidth(1.0);

  // Callback that updates the mesh positions of the 3D mesh
  int num_frames = V_hist.size();
  polyscope::state::userCallback = [&]() -> void {
    int i = silk::FramePlayer(num_frames);
    polyscope::registerSurfaceMesh("Triangulated Grid", V_hist[i], T);
  };

  polyscope::show();
}