#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <iostream>

using namespace std;

namespace silk {
Eigen::Vector3d make3D(Eigen::Vector2d v2);
tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeRegularTetrahedron();
tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeStackedTetrahedra();

}  // namespace silk