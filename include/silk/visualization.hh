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

}  // namespace silk