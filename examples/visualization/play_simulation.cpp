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

void registerMeshes(Eigen::MatrixXd &vertexPositions, Eigen::MatrixXi &triangles, Eigen::MatrixXi &tetrahedra) {
  polyscope::registerSurfaceMesh("Surfaces", vertexPositions, triangles);
  polyscope::registerTetMesh("Volumes", vertexPositions, tetrahedra);
}

tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeTriangle() {
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(3, 3);  // 3D vertex positions
  Eigen::RowVector3d v0(0.0, 0.0, 0.0);
  Eigen::RowVector3d v1(1.0, 0.0, 0.0);
  Eigen::RowVector3d v2(0.0, 1.0, 0.0);
  V << v0, v1, v2;

  Eigen::MatrixXi T(1, 3);  // triangles
  T << 0, 1, 2;

  return std::make_tuple(V, T);
};

tuple<Eigen::MatrixXd, Eigen::MatrixXi> combineTetrahedra(Eigen::MatrixXd &vertexPositions0,
                                                          Eigen::MatrixXd &vertexPositions1,
                                                          Eigen::MatrixXi &tetrahedra0,
                                                          Eigen::MatrixXi &tetrahedra1) {

  Eigen::MatrixXd V0 = vertexPositions0;
  Eigen::MatrixXd V1 = vertexPositions1;
  Eigen::MatrixXi T0 = tetrahedra0;
  Eigen::MatrixXi T1 = tetrahedra1;

  Eigen::MatrixXd vertexPositions(V0.rows() + V1.rows(), 3);
  vertexPositions << V0, V1;

  Eigen::MatrixXi tetrahedra(T0.rows() + T1.rows(), T0.cols());
  tetrahedra << T0, T1.array() + V0.rows();
  return std::make_tuple(vertexPositions, tetrahedra);
}

tuple<Eigen::MatrixXd, Eigen::MatrixXi> appendVertices(Eigen::MatrixXd &vertexPositions0,
                                                       Eigen::MatrixXd &vertexPositions1,
                                                       Eigen::MatrixXi &elements) {

  Eigen::MatrixXd V0 = vertexPositions0;
  Eigen::MatrixXd V1 = vertexPositions1;
  Eigen::MatrixXi E = elements;

  Eigen::MatrixXd vertexPositions(V0.rows() + V1.rows(), 3);
  vertexPositions << V0, V1;

  E = E.array() + V0.rows();  // Account for the offset caused by the vertices that were already present.
  return std::make_tuple(vertexPositions, E);
}

int main() {
  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;

  Eigen::MatrixXd vertexPositions;
  Eigen::MatrixXi triangles;
  Eigen::MatrixXi tetrahedra;

  Eigen::MatrixXd tetrahedronVertices0;
  Eigen::MatrixXi tetrahedra0;
  std::tie(tetrahedronVertices0, tetrahedra0) = silk::makeRegularTetrahedron();

  Eigen::MatrixXd tetrahedronVertices1;
  Eigen::MatrixXi tetrahedra1;
  std::tie(tetrahedronVertices1, tetrahedra1) = silk::makeRegularTetrahedron();
  tetrahedronVertices1.rightCols(1).array() += 1.5;  // Move tetrahedron up

  std::tie(vertexPositions,
           tetrahedra) = combineTetrahedra(tetrahedronVertices0, tetrahedronVertices1, tetrahedra0, tetrahedra1);

  Eigen::MatrixXd triangleVertices0;
  Eigen::MatrixXi triangles0;
  std::tie(triangleVertices0, triangles0) = makeTriangle();

  std::tie(vertexPositions, triangles) = appendVertices(vertexPositions, triangleVertices0, triangles0);

  std::cout << vertexPositions << std::endl;
  std::cout << triangles << std::endl;

  registerMeshes(vertexPositions, triangles, tetrahedra);

  //   polyscope::registerTetMesh("my mesh", V, T);

  //   // Add a scalar quantity
  //   size_t nVerts = V.rows();
  //   std::vector<double> scalarV(nVerts);
  //   for (size_t i = 0; i < nVerts; i++) {
  //     // use the z-coordinate of vertex position as a test function
  //     scalarV[i] = V(i, 2);
  //   }
  //   polyscope::getVolumeMesh("my mesh")->addVertexScalarQuantity("scalar Q", scalarV);

  // Show the GUI
  polyscope::show();
}