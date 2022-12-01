#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <igl/triangle/triangulate.h>

#include <TinyAD/ScalarFunction.hh>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

int main() {
  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;

  Eigen::MatrixXd V;
  Eigen::MatrixXi T;
  std::tie(V, T) = silk::makeRegularTetrahedron();

  // Pre-compute tetrahedral rest shapes
  std::vector<Eigen::Matrix3d> rest_shapes(T.rows());
  for (int t_idx = 0; t_idx < T.rows(); ++t_idx) {
    // Get 3D vertex positions
    Eigen::Vector3d ar = V.row(T(t_idx, 0));
    Eigen::Vector3d br = V.row(T(t_idx, 1));
    Eigen::Vector3d cr = V.row(T(t_idx, 2));
    Eigen::Vector3d dr = V.row(T(t_idx, 3));

    // Save 3-by-3 matrix with edge vectors as colums
    rest_shapes[t_idx] = TinyAD::col_mat(br - ar, cr - ar, dr - ar);
  };

  // Small squash
  V(0, 2) -= 0.2;

  // Set up a function with 3D vertex positions as variables
  auto tetrahedronEnergies = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

  tetrahedronEnergies.add_elements<4>(TinyAD::range(T.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    // Evaluate element using either double or TinyAD::Double
    using ScalarT = TINYAD_SCALAR_TYPE(element);
    int t_idx = element.handle;
    Eigen::Vector3<ScalarT> a = element.variables(T(t_idx, 0));
    Eigen::Vector3<ScalarT> b = element.variables(T(t_idx, 1));
    Eigen::Vector3<ScalarT> c = element.variables(T(t_idx, 2));
    Eigen::Vector3<ScalarT> d = element.variables(T(t_idx, 3));
    Eigen::Matrix3<ScalarT> M = TinyAD::col_mat(b - a, c - a, d - a);

    // Compute tet Jacobian and energy
    Eigen::Matrix3d Mr = rest_shapes[t_idx];
    Eigen::Matrix3<ScalarT> J = M * Mr.inverse();
    double vol = Mr.determinant() / 6.0;

    // std::cout << "J" << J << std::endl;

    // The exponential symmetric Dirichlet energy:
    return vol * exp((J.squaredNorm() + J.inverse().squaredNorm()));
  });

  Eigen::VectorXd positions = tetrahedronEnergies.x_from_data([&](int v_idx) { return V.row(v_idx); });

  auto [energy, gradient, projectedHessian] = tetrahedronEnergies.eval_with_hessian_proj(positions);
  std::cout << "Energy: " << energy << std::endl;
  Eigen::VectorXd forces = -gradient;

  Eigen::MatrixXd forcesMatrix = Eigen::MatrixXd::Zero(V.rows(), 3);
  tetrahedronEnergies.x_to_data(forces, [&](int v_idx, const Eigen::Vector3d &p) { forcesMatrix.row(v_idx) = p; });

  std::cout << "forces: " << forces << std::endl;

  polyscope::registerTetMesh("my mesh", V, T);
  auto vectorQ = polyscope::getVolumeMesh("my mesh")->addVertexVectorQuantity("forces", forcesMatrix);
  vectorQ->setEnabled(true);            // initially enabled
  vectorQ->setVectorLengthScale(0.05);  // make the vectors bigger

  // Show the GUI
  polyscope::show();
}