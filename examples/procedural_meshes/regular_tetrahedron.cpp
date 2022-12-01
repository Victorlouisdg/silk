#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <igl/triangle/triangulate.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

int main() {
  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(4, 3);  // 3D vertex positions

  // Adapted from:  https://www.danielsieger.com/blog/2021/01/03/generating-platonic-solids.html
  // choose coordinates on the unit sphere
  double a = 1.0 / 3.0;
  double b = sqrt(8.0 / 9.0);
  double c = sqrt(2.0 / 9.0);
  double d = sqrt(2.0 / 3.0);

  Eigen::RowVector3d v0(0, 0, 1);
  Eigen::RowVector3d v1(-c, d, -a);
  Eigen::RowVector3d v2(-c, -d, -a);
  Eigen::RowVector3d v3(b, 0, -a);
  V << v0, v1, v2, v3;

  Eigen::MatrixXi T(1, 4);  // tetrahedra
  T << 0, 1, 2, 3;

  polyscope::registerTetMesh("my mesh", V, T);

  // Add a scalar quantity
  size_t nVerts = V.rows();
  std::vector<double> scalarV(nVerts);
  for (size_t i = 0; i < nVerts; i++) {
    // use the z-coordinate of vertex position as a test function
    scalarV[i] = V(i, 2);
  }
  polyscope::getVolumeMesh("my mesh")->addVertexScalarQuantity("scalar Q", scalarV);

  // Show the GUI
  polyscope::show();
}