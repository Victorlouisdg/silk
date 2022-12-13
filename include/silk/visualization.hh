#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/curve_network.h"
#include "polyscope/point_cloud.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace silk {

namespace state {
int playback_frame_counter = 0;
bool playback_paused = false;
}  // namespace state

void registerInPolyscope(Eigen::Matrix<double, Eigen::Dynamic, 3> &vertexPositions,
                         vector<Eigen::ArrayXi> &points,
                         vector<Eigen::ArrayX2i> &edges,
                         vector<Eigen::ArrayX3i> &triangles,
                         vector<Eigen::ArrayX4i> &tetrahedra) {
  // TODO: Currently all vertexPositions are added to all Polyscope structure. However this might beneficial consider
  // whether it could be beneficial to extract only the vertexPositions used for each structure. This could improve
  // performance and would allow custom vertexQuantities to be added to each structure separately. However it would
  // require careful reindexing. Currently the vertexQuantities are global, so we add them only once to a sinlge point
  // cloud.
  polyscope::registerPointCloud("Vertices", vertexPositions);

  for (Eigen::ArrayXi pointGroup : points) {
    polyscope::registerPointCloud("Points", vertexPositions(pointGroup, Eigen::all));
  }

  for (Eigen::ArrayX2i edgeGroup : edges) {
    polyscope::registerCurveNetwork("Edges", vertexPositions, edgeGroup);
  }

  for (Eigen::ArrayX3i triangleGroup : triangles) {
    polyscope::registerSurfaceMesh("Surfaces", vertexPositions, triangleGroup);
  }

  for (Eigen::ArrayX4i tetrahedronGroup : tetrahedra) {
    polyscope::registerTetMesh("Volumes", vertexPositions, tetrahedronGroup);
  }
}

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