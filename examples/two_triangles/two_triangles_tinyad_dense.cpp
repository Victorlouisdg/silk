#include <TinyAD/Scalar.hh>

#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

void visualize(std::vector<std::vector<size_t>> triangles, std::vector<Vector3> vertexCoordinates) {
  SimplePolygonMesh simpleMesh(triangles, vertexCoordinates);

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeManifoldSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry->vertexPositions, mesh->getFaceVertexList());
  // addFaceNormals(*geometry, *mesh, *psMesh);
  polyscope::show();
}

int main() {
  // Choose autodiff scalar type for 3 variables
  using ADouble = TinyAD::Double<3>;
  // Init a 3D vector of active variables and a 3D vector of passive variables
  Eigen::Vector3<ADouble> x0{0.0, 0.0, 0.0};  //  = ADouble::make_active({0.0, 0.0, 0.0});
  Eigen::Vector3<ADouble> x1{1.0, 0.0, 0.0};  // = ADouble::make_active({1.0, 0.0, 0.0});
  // Eigen::Vector3<ADouble> x2 = ADouble::make_active({0.5, 0.5, 0.0});
  Eigen::Vector3<ADouble> x2 = ADouble::make_active({0.5, 1.0, 0.0});

  Eigen::Vector2<double> u0{0.0, 0.0};
  Eigen::Vector2<double> u1{1.0, 0.0};
  Eigen::Vector2<double> u2{0.5, 0.5};

  std::vector<Eigen::Vector3<ADouble>> positions{x0, x1, x2};
  std::vector<Eigen::Vector2<double>> positions_uv{u0, u1, u2};

  Eigen::Matrix<ADouble, 2, 2> Dm;
  Dm.block<2, 1>(0, 0) = u1 - u0;
  Dm.block<2, 1>(0, 1) = u2 - u0;

  Eigen::Matrix<ADouble, 3, 2> Ds;
  Ds.block<3, 1>(0, 0) = x1 - x0;
  Ds.block<3, 1>(0, 1) = x2 - x0;

  Eigen::Matrix<ADouble, 2, 2> Dm_inv = Dm.inverse();

  std::cout << TinyAD::to_passive(Dm) << std::endl;
  std::cout << TinyAD::to_passive(Dm_inv) << std::endl;

  Eigen::Matrix<ADouble, 3, 2> F = Ds * Dm_inv;

  std::cout << TinyAD::to_passive(F) << std::endl;

  Eigen::Matrix<ADouble, 3, 1> wu = F.block<3, 1>(0, 0);
  Eigen::Matrix<ADouble, 3, 1> wv = F.block<3, 1>(0, 1);

  ADouble Cu = wu.norm() - 1.0;
  ADouble Cv = wv.norm() - 1.0;

  ADouble Eu = 0.5 * Cu * Cu;
  ADouble Ev = 0.5 * Cv * Cv;
  ADouble E = Eu + Ev;

  std::cout << "Energy:" << TinyAD::to_passive(E) << std::endl;
  std::cout << "Force on x1 (gradient of the energy):\n" << TinyAD::to_passive(E.grad) << std::endl;
  std::cout << "Hessian of the energy:\n" << TinyAD::to_passive(E.Hess) << std::endl;
}