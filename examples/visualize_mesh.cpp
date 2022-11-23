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

void visualizeMesh(std::string meshPath) {
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = readManifoldSurfaceMesh(meshPath);

  polyscope::init();

  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry->vertexPositions, mesh->getFaceVertexList());

  addSmoothestVertexDirectionField(*geometry, *mesh, *psMesh);
  addFaceAreas(*geometry, *mesh, *psMesh);
  addFaceNormals(*geometry, *mesh, *psMesh);
  addCurvatures(*geometry, *mesh, *psMesh);
  addEdgeDihedralAngle(*geometry, *mesh, *psMesh);

  polyscope::show();
}

int main(int argc, char *argv[]) {
  std::string meshPath = argv[1];
  visualizeMesh(meshPath);
}