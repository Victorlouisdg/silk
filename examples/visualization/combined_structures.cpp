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

#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <igl/triangle/triangulate.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

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

int main() {
  Eigen::Matrix<double, Eigen::Dynamic, 3> vertexPositions;

  // Add the spring
  Eigen::RowVector3d v0(0.0, 0.0, 3.0);
  Eigen::RowVector3d v1(0.0, 0.0, 2.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v0, v1;
  Eigen::ArrayX2i springEdge(1, 2);
  springEdge << 0, 1;

  // Add the triangle
  Eigen::RowVector3d v2(1.0, 0.0, 1.0);
  Eigen::RowVector3d v3(-1.0, 0.0, 1.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v2, v3;
  Eigen::ArrayX3i triangle(1, 3);
  triangle << 1, 2, 3;

  // Add the tetrahedron
  Eigen::RowVector3d v4(0.0, 1.0, 0.0);
  Eigen::RowVector3d v5(0.0, -1.0, 0.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 2, 3);
  vertexPositions.bottomRows(2) << v4, v5;
  Eigen::ArrayX4i tetrahedron(1, 4);
  tetrahedron << 2, 3, 4, 5;

  // Add an extra point
  Eigen::RowVector3d v6(0.0, 0.0, 4.0);
  vertexPositions.conservativeResize(vertexPositions.rows() + 1, 3);
  vertexPositions.bottomRows(1) << v6;
  Eigen::ArrayXi points_(3);
  points_ << 0, 1, 6;

  vector<Eigen::ArrayXi> points{points_};
  vector<Eigen::ArrayX2i> edges{springEdge};
  vector<Eigen::ArrayX3i> triangles{triangle};
  vector<Eigen::ArrayX4i> tetrahedra{tetrahedron};

  std::cout << vertexPositions << std::endl;

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  registerInPolyscope(vertexPositions, points, edges, triangles, tetrahedra);
  polyscope::show();
}