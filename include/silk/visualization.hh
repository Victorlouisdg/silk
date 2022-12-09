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

namespace silk {
namespace state {
int playback_frame_counter = 0;
bool playback_paused = false;
}  // namespace state

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
  psMesh.addFaceScalarQuantity("areas", geometry.faceAreas);
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

void visualizeMesh(SurfaceMesh &mesh, VertexPositionGeometry &geometry) {
  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry.vertexPositions, mesh.getFaceVertexList());

  // addSmoothestVertexDirectionField(geometry, mesh, *psMesh);
  // addFaceAreas(geometry, mesh, *psMesh);
  addFaceNormals(geometry, mesh, *psMesh);
  // addCurvatures(geometry, mesh, *psMesh);
  // addEdgeDihedralAngle(geometry, mesh, *psMesh);

  polyscope::show();
}
}  // namespace silk