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

#include <igl/edges.h>
#include <igl/triangle/triangulate.h>

#include <ipc/barrier/barrier.hpp>
#include <ipc/ipc.hpp>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

int main() {

  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(4, 3);  // 3D vertex positions

  Eigen::RowVector3d v0(0, 0, 0);
  Eigen::RowVector3d v2(1, 0, 0);
  Eigen::RowVector3d v1(0, 1, 0);
  Eigen::RowVector3d v3(0.2, 0.2, 0.2);
  V << v0, v1, v2, v3;

  Eigen::MatrixXi F(1, 3);  // tetrahedra
  F << 0, 1, 2;

  Eigen::MatrixXi E;
  igl::edges(F, E);

  // Eigen::MatrixXi E(3, 2);  // tetrahedra
  // E << 0, 1, 1, 2, 2, 0;

  ipc::FaceVertexCandidate candidate(0, 3);

  ipc::PointTriangleDistanceType distanceType = ipc::PointTriangleDistanceType::AUTO;
  double distance = sqrt(candidate.compute_distance(V, E, F, distanceType));
  ipc::VectorMax12d gradient = candidate.compute_distance_gradient(V, E, F, distanceType);
  ipc::MatrixMax12d hessian = candidate.compute_distance_hessian(V, E, F, distanceType);
  std::cout << "Distance: " << distance << std::endl;
  std::cout << "Gradient: " << std::endl;
  std::cout << gradient << std::endl;
  std::cout << "Hessian: " << std::endl;
  std::cout << hessian << std::endl;

  double dhat = 1.0;  // maximum distance at which repulsion works

  double B = ipc::barrier<double>(distance, dhat);
  double gradB = ipc::barrier_gradient(distance, dhat);
  double hessianB = ipc::barrier_hessian(distance, dhat);
  std::cout << "Barrier potential: " << B << std::endl;
  std::cout << "Barrier gradient: " << gradB << std::endl;
  std::cout << "Barrier hessian: " << hessianB << std::endl;

  // Constructor when all vertices are known to be on the surface of the mesh.
  ipc::CollisionMesh mesh(V, E, F);

  // This constructor removes the free floating codimensional point.
  // ipc::CollisionMesh mesh;
  // mesh = ipc::CollisionMesh::build_from_full_mesh(V, E, F);
  Eigen::MatrixXd collisionV = mesh.vertices(V);

  std::cout << collisionV << std::endl;

  ipc::BroadPhaseMethod method = ipc::BroadPhaseMethod::BRUTE_FORCE;
  ipc::Constraints constraint_set;
  constraint_set.build(mesh, collisionV, dhat, /*dmin=*/0, method);

  Eigen::VectorXd grad_b = ipc::compute_barrier_potential_gradient(mesh, collisionV, constraint_set, dhat);
  std::cout << "Barrier gradient: " << grad_b << std::endl;

  Eigen::SparseMatrix<double> hess_b = ipc::compute_barrier_potential_hessian(
      mesh, V, constraint_set, dhat, /*project_to_psd=*/false);

  std::cout << "Barrier hessian: " << std::endl << hess_b << std::endl;
}