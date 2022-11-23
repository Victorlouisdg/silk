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

#include <TinyAD-Examples/IGLGeometryCentralConvert.hh>

#include "silk/energies.hh"
#include "silk/simple_meshes.hh"

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

geometrycentral::Vector3 to_geometrycentral(const Eigen::Vector3d &_v) {
  return geometrycentral::Vector3{_v.x(), _v.y(), _v.z()};
}

FaceData<Eigen::Matrix2d> computeDm_invs(ManifoldSurfaceMesh &mesh,
                                         VertexPositionGeometry &geometry,
                                         Eigen::MatrixXd &P) {
  FaceData<Eigen::Matrix2d> Dm_invs(mesh);

  geometry.requireVertexIndices();

  for (auto f : mesh.faces()) {
    // Get 3D vertex positions

    Vertex v0 = f.halfedge().vertex();
    Vertex v1 = f.halfedge().next().vertex();
    Vertex v2 = f.halfedge().next().next().vertex();

    int i = geometry.vertexIndices[v0];
    int j = geometry.vertexIndices[v1];
    int k = geometry.vertexIndices[v2];

    Eigen::Vector2d u0 = P.row(i);
    Eigen::Vector2d u1 = P.row(j);
    Eigen::Vector2d u2 = P.row(k);

    Eigen::Matrix2d Dm;
    Dm.col(0) = u1 - u0;
    Dm.col(1) = u2 - u0;

    Eigen::Matrix2d Dm_inv = Dm.inverse();
    Dm_invs[f] = Dm_inv;
  };
  return Dm_invs;
}

int main() {
  unique_ptr<ManifoldSurfaceMesh> mesh;
  unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = silk::makeTwoTriangleSquare();

  Eigen::MatrixXd V;               // 3D vertex positions
  Eigen::MatrixXi F;               // Mesh faces
  to_igl(*mesh, *geometry, V, F);  // Convert mesh to igl format

  Eigen::MatrixXd P = V.leftCols<2>();  // 2D parametrization positions
  P.row(3) = Eigen::Vector2d(0.0, 1.0);

  geometry->requireVertexIndices();
  FaceData<Eigen::Matrix2d> Dm_invs = computeDm_invs(*mesh, *geometry, P);

  // Set up a function with 3D vertex positions as variables
  auto meshEnergyFunction = TinyAD::scalar_function<3>(mesh->vertices());

  // Add objective term per face. Each connecting 3 vertices.
  meshEnergyFunction.add_elements<3>(mesh->faces(), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    // Evaluate element using either double or TinyAD::Double
    using T = TINYAD_SCALAR_TYPE(element);

    // Get variable 2D vertex positions
    Face f = element.handle;
    Vertex v0 = f.halfedge().vertex();
    Vertex v1 = f.halfedge().next().vertex();
    Vertex v2 = f.halfedge().next().next().vertex();
    Eigen::Vector3<T> x0 = element.variables(v0);
    Eigen::Vector3<T> x1 = element.variables(v1);
    Eigen::Vector3<T> x2 = element.variables(v2);

    int i = geometry->vertexIndices[v0];
    int j = geometry->vertexIndices[v1];
    int k = geometry->vertexIndices[v2];

    Eigen::Vector2d u0 = P.row(i);
    Eigen::Vector2d u1 = P.row(j);
    Eigen::Vector2d u2 = P.row(k);

    Eigen::Matrix<T, 3, 2> F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
    T E = silk::baraffWitkinStretchEnergy(F);
    return E;
  });

  geometry->requireVertexPositions();
  Eigen::VectorXd x = meshEnergyFunction.x_from_data([&](Vertex v) { return to_eigen(geometry->vertexPositions[v]); });

  auto [E, g, H_proj] = meshEnergyFunction.eval_with_hessian_proj(x);

  Eigen::VectorXd f0 = g;
  Eigen::SparseMatrix<double> dfdx = H_proj;

  int amount_of_vertices = geometry->vertexPositions.size();
  int system_size = 3 * amount_of_vertices;

  Eigen::VectorXd masses = Eigen::VectorXd::Ones(system_size);

  std::cout << masses << std::endl;

  Eigen::VectorXd accelerations = f0.array() / masses.array();
  Eigen::VectorXd velocities = Eigen::VectorXd::Zero(system_size);
  double dt = 0.01;
  velocities += accelerations * dt;

  Eigen::VectorXd positions = x;
  positions += velocities * dt;

  std::cout << "x:\n" << x << std::endl;
  std::cout << "positions:\n" << positions << std::endl;

  Eigen::SparseMatrix<double> identity_matrix(system_size, system_size);
  identity_matrix.setIdentity();
}