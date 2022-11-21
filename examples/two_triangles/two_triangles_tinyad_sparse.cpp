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

using namespace geometrycentral;
using namespace geometrycentral::surface;

geometrycentral::Vector3 to_geometrycentral(const Eigen::Vector3d &_v) {
  return geometrycentral::Vector3{_v.x(), _v.y(), _v.z()};
}

template<typename T>
T baraffWitkinStretchEnergy(Eigen::Vector3<T> x0, Eigen::Vector3<T> x1, Eigen::Vector3<T> x2, Eigen::Matrix2d Dm_inv) {
  Eigen::Matrix<T, 3, 2> Ds;
  Ds.col(0) = x1 - x0;
  Ds.col(1) = x2 - x0;
  Eigen::Matrix<T, 3, 2> F = Ds * Dm_inv;

  Eigen::Matrix<T, 3, 1> wu = F.col(0);
  Eigen::Matrix<T, 3, 1> wv = F.col(1);

  T Cu = wu.norm() - 1.0;
  T Cv = wv.norm() - 1.0;

  T Eu = 0.5 * Cu * Cu;
  T Ev = 0.5 * Cv * Cv;
  T E = Eu + Ev;
  return E;
}

int main() {
  // Setup mesh and rest shape
  std::vector<size_t> triangle0{0, 1, 2};
  std::vector<size_t> triangle1{0, 2, 3};
  std::vector<std::vector<size_t>> triangles{triangle0, triangle1};

  Vector3 v0{0.0, 0.0, 0.0};
  Vector3 v1{1.0, 0.0, 0.0};
  Vector3 v2{1.0, 1.0, 0.0};
  Vector3 v3{0.0, 1.0, 0.0};
  Vector3 v4{0.0, 2.0, 0.0};

  std::vector<Vector3> vertexCoordinates{v0, v1, v2, v3};
  std::vector<Vector3> vertexCoordinates2{v0, v1, v2, v4};

  SimplePolygonMesh simpleMesh(triangles, vertexCoordinates2);

  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeManifoldSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);

  Eigen::MatrixXd V;               // 3D vertex positions
  Eigen::MatrixXi F;               // Mesh faces
  to_igl(*mesh, *geometry, V, F);  // Convert mesh to igl format

  Eigen::MatrixXd P = V.leftCols<2>();  // 2D parametrization positions
  P.row(3) = Eigen::Vector2d(0.0, 1.0);

  FaceData<Eigen::Matrix2d> Dm_invs(*mesh);

  geometry->requireVertexIndices();

  for (auto f : mesh->faces()) {
    // Get 3D vertex positions

    Vertex v0 = f.halfedge().vertex();
    Vertex v1 = f.halfedge().next().vertex();
    Vertex v2 = f.halfedge().next().next().vertex();

    int i = geometry->vertexIndices[v0];
    int j = geometry->vertexIndices[v1];
    int k = geometry->vertexIndices[v2];

    Eigen::Vector2d u0 = P.row(i);
    Eigen::Vector2d u1 = P.row(j);
    Eigen::Vector2d u2 = P.row(k);

    Eigen::Matrix2d Dm;
    Dm.col(0) = u1 - u0;
    Dm.col(1) = u2 - u0;

    Eigen::Matrix2d Dm_inv = Dm.inverse();
    Dm_invs[f] = Dm_inv;
  };

  // Set up a function with 3D vertex positions as variables
  auto energy_func = TinyAD::scalar_function<3>(mesh->vertices());

  // Add objective term per face. Each connecting 3 vertices.
  energy_func.add_elements<3>(mesh->faces(), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    // Evaluate element using either double or TinyAD::Double
    using T = TINYAD_SCALAR_TYPE(element);

    // Get variable 2D vertex positions
    Face f = element.handle;
    Eigen::Vector3<T> x0 = element.variables(f.halfedge().vertex());
    Eigen::Vector3<T> x1 = element.variables(f.halfedge().next().vertex());
    Eigen::Vector3<T> x2 = element.variables(f.halfedge().next().next().vertex());

    T E = baraffWitkinStretchEnergy(x0, x1, x2, Dm_invs[f]);
    return E;
  });

  Face f = *(mesh->faces().begin());
  Eigen::Vector3d x0 = to_eigen(geometry->vertexPositions[f.halfedge().vertex()]);
  Eigen::Vector3d x1 = to_eigen(geometry->vertexPositions[f.halfedge().next().vertex()]);
  Eigen::Vector3d x2 = to_eigen(geometry->vertexPositions[f.halfedge().next().next().vertex()]);
  double single_triangle_E = baraffWitkinStretchEnergy(x0, x1, x2, Dm_invs[f]);
  std::cout << "Energy single:" << single_triangle_E << std::endl;

  geometry->requireVertexPositions();
  Eigen::VectorXd x = energy_func.x_from_data([&](Vertex v) { return to_eigen(geometry->vertexPositions[v]); });

  double E0 = energy_func.eval(x);
  std::cout << "Energy:" << E0 << std::endl;

  auto [E1, g, H_proj] = energy_func.eval_with_hessian_proj(x);

  std::cout << "Energy:" << TinyAD::to_passive(E1) << std::endl;
  std::cout << "Force on x1 (gradient of the energy):\n" << TinyAD::to_passive(g) << std::endl;
  std::cout << "Hessian of the energy:\n" << TinyAD::to_passive(H_proj) << std::endl;

  VertexData<Vector3> gradients(*mesh);

  energy_func.x_to_data(g, [&](Vertex v, const Eigen::Vector3d &p) { gradients[v] = to_geometrycentral(p); });

  // Visualization
  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry->vertexPositions, mesh->getFaceVertexList());
  geometry->requireFaceNormals();
  psMesh->addFaceVectorQuantity("normals", geometry->faceNormals);
  psMesh->addVertexVectorQuantity("gradients", gradients);
  polyscope::show();
}