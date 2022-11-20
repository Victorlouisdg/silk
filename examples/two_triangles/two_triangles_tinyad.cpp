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
  Eigen::Vector3<ADouble> x = ADouble::make_active({0.0, -1.0, 1.0});
  Eigen::Vector3<double> y(2.0, 3.0, 5.0);

  // Compute angle using Eigen functions and retrieve gradient and Hessian w.r.t. x
  ADouble angle = acos(x.dot(y) / (x.norm() * y.norm()));
  Eigen::Vector3d g = angle.grad;
  Eigen::Matrix3d H = angle.Hess;

  std::cout << angle << std::endl;
  std::cout << g << std::endl;
  std::cout << H << std::endl;
}