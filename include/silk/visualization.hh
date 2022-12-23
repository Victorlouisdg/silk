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
                         vector<Eigen::ArrayXi> &pointGroups,
                         vector<Eigen::ArrayX2i> &edgeGroups,
                         vector<Eigen::ArrayX3i> &triangleGroups,
                         vector<Eigen::ArrayX4i> &tetrahedronGroups,
                         const map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>> &vertexVectorQuantities =
                             map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>()) {
  // TODO: Currently all vertexPositions are added to all Polyscope structure. However this might beneficial consider
  // whether it could be beneficial to extract only the vertexPositions used for each structure. This could improve
  // performance and would allow custom vertexQuantities to be added to each structure separately. However it would
  // require careful reindexing. Currently the vertexQuantities are global, so we add them only once to a sinlge point
  // cloud.
  polyscope::PointCloud *psCloud = polyscope::registerPointCloud("Vertices", vertexPositions);
  // for (auto &vertexVectorQuantity : vertexVectorQuantities) {
  //   psCloud->addVectorQuantity(vertexVectorQuantity.first, vertexVectorQuantity.second);
  // }
  for (auto const &[quantityName, quantity] : vertexVectorQuantities) {
    psCloud->addVectorQuantity(quantityName, quantity);
  }

  // psCloud->addVectorQuantity2D()

  for (int i = 0; i < pointGroups.size(); i++) {
    polyscope::registerPointCloud("Point group " + std::to_string(i), vertexPositions(pointGroups[i], Eigen::all));
  }

  for (int i = 0; i < edgeGroups.size(); i++) {
    polyscope::registerCurveNetwork("Edges group " + std::to_string(i), vertexPositions, edgeGroups[i]);
  }

  for (int i = 0; i < triangleGroups.size(); i++) {
    polyscope::registerSurfaceMesh("Triangle group " + std::to_string(i), vertexPositions, triangleGroups[i]);
  }

  for (int i = 0; i < tetrahedronGroups.size(); i++) {
    polyscope::registerTetMesh("Tetrahedron group " + std::to_string(i), vertexPositions, tetrahedronGroups[i]);
  }
}

void playHistoryCallback(vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> vertexPositionHistory,
                         vector<Eigen::ArrayXi> pointGroups,
                         vector<Eigen::ArrayX2i> edgeGroups,
                         vector<Eigen::ArrayX3i> triangleGroups,
                         vector<Eigen::ArrayX4i> tetrahedraGroups,
                         vector<map<string, Eigen::Matrix<double, Eigen::Dynamic, 3>>> vertexVectorQuantitiesHistory) {

  int frame = silk::state::playback_frame_counter % vertexPositionHistory.size();
  ImGui::Text("Frame %d", frame);

  if (ImGui ::Button("Next frame")) {
    silk::state::playback_frame_counter++;
    frame = silk::state::playback_frame_counter % vertexPositionHistory.size();
    silk::registerInPolyscope(vertexPositionHistory[frame],
                              pointGroups,
                              edgeGroups,
                              triangleGroups,
                              tetrahedraGroups,
                              vertexVectorQuantitiesHistory[frame]);
  }

  if (ImGui ::Button("Previous frame")) {
    silk::state::playback_frame_counter--;
    frame = silk::state::playback_frame_counter % vertexPositionHistory.size();
    silk::registerInPolyscope(vertexPositionHistory[frame],
                              pointGroups,
                              edgeGroups,
                              triangleGroups,
                              tetrahedraGroups,
                              vertexVectorQuantitiesHistory[frame]);
  }

  // Play-pause logic
  if (silk::state::playback_paused) {
    if (ImGui ::Button("Resume playback")) {
      silk::state::playback_paused = false;
    }
  } else {
    if (ImGui::Button("Pause playback")) {
      silk::state::playback_paused = true;
    }
  }
  if (silk::state::playback_paused) {
    return;
  }

  silk::registerInPolyscope(vertexPositionHistory[frame],
                            pointGroups,
                            edgeGroups,
                            triangleGroups,
                            tetrahedraGroups,
                            vertexVectorQuantitiesHistory[frame]);
  silk::state::playback_frame_counter++;
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