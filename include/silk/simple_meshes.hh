#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <iostream>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace silk {
Eigen::Vector3d make3D(Eigen::Vector2d v2);
tuple<unique_ptr<ManifoldSurfaceMesh>, unique_ptr<VertexPositionGeometry>> makeSingleTriangle();
tuple<unique_ptr<ManifoldSurfaceMesh>, unique_ptr<VertexPositionGeometry>> makeTwoTriangleSquare();
tuple<unique_ptr<ManifoldSurfaceMesh>, unique_ptr<VertexPositionGeometry>> makeSquare();
VertexData<Eigen::Vector2d> makeProjectedRestPositions(ManifoldSurfaceMesh &mesh, VertexPositionGeometry &geometry);
tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeRegularTetrahedron();
tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeStackedTetrahedra();

}  // namespace silk