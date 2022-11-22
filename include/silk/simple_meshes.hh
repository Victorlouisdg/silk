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

tuple<unique_ptr<ManifoldSurfaceMesh>, unique_ptr<VertexPositionGeometry>> makeTwoTriangleSquare();

}  // namespace silk