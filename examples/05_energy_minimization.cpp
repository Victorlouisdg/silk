#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <Eigen/Core>
#include <igl/doublearea.h>
#include <igl/triangulated_grid.h>

#include "silk/deformation/deformation_gradient.hh"
#include "silk/deformation/rest_shape.hh"
#include "silk/energy/baraff_witkin.hh"
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

  // Compute the deformation gradients
  std::vector<Eigen::Matrix<double, 3, 2>> F = silk::deformation_gradients(V, T, V2);
  Eigen::VectorXd A2;  // area in 2D of rest shape triangles
  igl::doublearea(V2, T, A2);
  A2 = 0.5 * A2.array();

  // Created the rest shape matrices and inverted them
  std::vector<Eigen::Matrix2d> Dm = silk::rest_shapes(V2, T);
  std::vector<Eigen::Matrix2d> Dm_inv = silk::inverse(Dm);

  Eigen::VectorXd K = Eigen::VectorXd::Constant(T.rows(), 1.0);  // Stretch stiffness for each triangle

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
    double a = A2(i); // area of the rest shape triangle
    double k = K(i);  // stretch stiffness of the triangle
    ST E = k * silk::stretch_bw(F, a);  // stretch energy of the triangle
    return E;
  });

  Eigen::VectorXd x = silk::flatten(V);
  auto [f, g, H_proj] = func.eval_with_hessian_proj(x);

  Eigen::MatrixXd forces = -silk::unflatten(g);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *mesh3D = polyscope::registerSurfaceMesh("Triangulated Grid", V, T);
  mesh3D->addVertexVectorQuantity("Forces", forces);
  mesh3D->setEdgeWidth(1.0);

  auto *mesh2D = polyscope::registerSurfaceMesh2D("Triangulated Grid 2D", V2, T);
  mesh2D->setTransparency(0.5);
  mesh2D->setEdgeWidth(1.0);
  polyscope::show();
}