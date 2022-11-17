#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>
#include <torch/torch.h>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

using namespace torch::indexing;

void addSmoothestVertexDirectionField(VertexPositionGeometry &geometry,
                                      SurfaceMesh &mesh,
                                      polyscope::SurfaceMesh &psMesh) {
  VertexData<Vector2> field = computeSmoothestVertexDirectionField(geometry);
  geometry.requireVertexTangentBasis();
  VertexData<Vector3> vBasisX(mesh);
  for (Vertex v : mesh.vertices()) {
    vBasisX[v] = geometry.vertexTangentBasis[v][0];
  }
  psMesh.setVertexTangentBasisX(vBasisX);
  psMesh.addVertexIntrinsicVectorQuantity("vectors", field);
}

void addFaceAreas(VertexPositionGeometry &geometry, SurfaceMesh &mesh, polyscope::SurfaceMesh &psMesh) {
  geometry.requireFaceAreas();

  FaceData<double> areas(mesh);
  int i = 0;
  for (Face f : mesh.faces()) {
    areas[i] = geometry.faceAreas[f];
    i++;
  }
  psMesh.addFaceScalarQuantity("areas", areas);
}

void addFaceNormals(VertexPositionGeometry &geometry, SurfaceMesh &mesh, polyscope::SurfaceMesh &psMesh) {
  geometry.requireFaceNormals();
  psMesh.addFaceVectorQuantity("normals", geometry.faceNormals);
}

void addCurvatures(VertexPositionGeometry &geometry, SurfaceMesh &mesh, polyscope::SurfaceMesh &psMesh) {
  geometry.requireVertexGaussianCurvatures();
  geometry.requireVertexMeanCurvatures();
  geometry.requireVertexMinPrincipalCurvatures();
  geometry.requireVertexMaxPrincipalCurvatures();

  psMesh.addVertexScalarQuantity("gaussian_curvatures", geometry.vertexGaussianCurvatures);
  psMesh.addVertexScalarQuantity("mean_curvatures", geometry.vertexMeanCurvatures);
  psMesh.addVertexScalarQuantity("min_principal_curvatures", geometry.vertexMinPrincipalCurvatures);
  psMesh.addVertexScalarQuantity("max_principal_curvatures", geometry.vertexMaxPrincipalCurvatures);
}

/*Dihedral edge*/

void addEdgeDihedralAngle(VertexPositionGeometry &geometry, SurfaceMesh &mesh, polyscope::SurfaceMesh &psMesh) {
  geometry.requireEdgeDihedralAngles();
  psMesh.addEdgeScalarQuantity("dihedral_angles", geometry.edgeDihedralAngles);
}

void visualizeMesh() {
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = readManifoldSurfaceMesh("../meshes/shirt_wrinkled.obj");

  polyscope::init();

  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry->vertexPositions, mesh->getFaceVertexList());

  addSmoothestVertexDirectionField(*geometry, *mesh, *psMesh);
  addFaceAreas(*geometry, *mesh, *psMesh);
  addFaceNormals(*geometry, *mesh, *psMesh);
  addCurvatures(*geometry, *mesh, *psMesh);
  addEdgeDihedralAngle(*geometry, *mesh, *psMesh);

  polyscope::show();
}

torch::Tensor deformationGradient(torch::Tensor positions, torch::Tensor positions_uv) {
  // Read out the rows. 
  torch::Tensor u0 = positions_uv.index({0, Slice()});
  torch::Tensor u1 = positions_uv.index({1, Slice()});
  torch::Tensor u2 = positions_uv.index({2, Slice()});
  return u0;
}

int main() {
  std::vector<size_t> triangle{0, 1, 2};
  std::vector<std::vector<size_t>> triangles{triangle};

  Vector3 v0{0.0, 0.0, 0.0};
  Vector3 v1{1.0, 0.0, 0.0};
  Vector3 v2{0.5, 0.5, 0.0};

  torch::Tensor positions = torch::zeros({3, 3});
  positions.index_put_({1, 0}, 1.0);
  positions.index_put_({2, 0}, 0.5);
  positions.index_put_({2, 1}, 0.5);
  std::cout << positions << std::endl;


  torch::Tensor positions_uv = positions.index({Slice(), Slice(None, 2)}); // positions[:, :2];
  std::cout << positions_uv << std::endl;

  torch::Tensor deformationGradient_ = deformationGradient(positions, positions_uv);
  std::cout << deformationGradient_ << std::endl;

  std::vector<Vector3> vertexCoordinates{v0, v1, v2};

  SimplePolygonMesh simpleMesh(triangles, vertexCoordinates);

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeManifoldSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);

  // polyscope::init();
  // polyscope::view::upDir = polyscope::UpDir::ZUp;
  // auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry->vertexPositions, mesh->getFaceVertexList());
  // addFaceNormals(*geometry, *mesh, *psMesh);
  // polyscope::show();
}