#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "silk/meshes/right_triangle.hh"
#include <Eigen/Core>
#include <igl/doublearea.h>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include <Eigen/Core>
#include <igl/doublearea.h>
#include <igl/triangulated_grid.h>

#include "silk/deformation/deformation_gradient.hh"
#include "silk/deformation/rest_shape.hh"
#include "silk/energy/baraff_witkin.hh"
#include "silk/math/inverse.hh"
#include "silk/meshes/right_triangle.hh"

#include "happly.h"

#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/rich_surface_mesh_data.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
using namespace geometrycentral::surface;

void calculate_mesh_and_attributes(Eigen::MatrixXd &V,
                                   Eigen::MatrixXd &V2,
                                   Eigen::MatrixXi &T,
                                   Eigen::VectorXd &A,
                                   Eigen::VectorXd &A2,
                                   Eigen::VectorXd &E) {
  // Create a triangulated grid in 2D
  igl::triangulated_grid(4, 4, V2, T);

  // Add a z coordinate to make the triangulated grid 3D
  V = Eigen::MatrixXd::Zero(V2.rows(), 3);
  V.leftCols(2) = V2;

  // Option 3. Quadratic deformation: z = x^2
  V.col(2) = V.col(0).array().square();

  // Compute the deformation gradients
  std::vector<Eigen::Matrix<double, 3, 2>> F = silk::deformation_gradients(V, T, V2);
  igl::doublearea(V2, T, A2);
  A2 = 0.5 * A2.array();

  // Compute an elastic potential energy for each triangle
  E = Eigen::VectorXd::Zero(T.rows());
  for (int i = 0; i < T.rows(); i++) {
    E(i) = silk::stretch_bw(F[i], A2[i]);
  }

  // Compute the areas in 3D of the deformed triangles
  igl::doublearea(V, T, A);
  A = 0.5 * A.array();
}

int main() {
  // Create a triangulated grid in 2D

  Eigen::MatrixXd V;   // Vertex positions in 3D
  Eigen::MatrixXd V2;  // Vertex positions in 2D
  Eigen::MatrixXi T;   // Triangles, not we abbreviate as T here to reserve F for the deformeation gradient
  Eigen::VectorXd A;   // Triangle areas
  Eigen::VectorXd A2;  // Triangle areas in 2D
  Eigen::VectorXd E;   // Elastic potential energy

  calculate_mesh_and_attributes(V, V2, T, A, A2, E);

  // Create geometry central data structures to save to ply
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeSurfaceMeshAndGeometry(V, T);

  // Testing with .obj first
  writeSurfaceMesh(*mesh, *geometry, "my_mesh.obj");

  FaceData<double> elasticEnergy(*mesh, E);

  RichSurfaceMeshData richData(*mesh);
  richData.addMeshConnectivity();
  richData.addGeometry(*geometry);
  richData.addFaceProperty("elasticEnergy", elasticEnergy);

  richData.outputFormat = happly::DataFormat::ASCII;
  richData.write("my_mesh.ply");

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *mesh3D = polyscope::registerSurfaceMesh("Triangulated Grid", V, T);
  mesh3D->addFaceScalarQuantity("E", E);
  mesh3D->addFaceScalarQuantity("A", A);
  mesh3D->setEdgeWidth(1.0);

  auto *mesh2D = polyscope::registerSurfaceMesh2D("Triangulated Grid 2D", V2, T);
  mesh2D->setTransparency(0.5);
  mesh2D->setEdgeWidth(1.0);
  polyscope::show();
}