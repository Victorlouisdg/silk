#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "silk/conversions.hh"
#include "silk/deformation/rest_shapes.hh"
#include "silk/energies.hh"
#include "silk/energy/energy_function.hh"
#include "silk/energy/energy_mapping.hh"
#include "silk/geometry/area.hh"
#include "silk/mesh_construction.hh"
#include "silk/optimization/line_search.hh"
#include "silk/simple_meshes.hh"
#include "silk/types.hh"
#include "silk/visualization.hh"

#include <igl/edges.h>
#include <igl/triangle/triangulate.h>

#include <ipc/friction/friction.hpp>
#include <ipc/ipc.hpp>

#include <Eigen/CholmodSupport>

using namespace std;
using namespace std::chrono;

int main() {
  VertexPositions vertexPositions;
  vector<Points> pointGroups;
  vector<Edges> edgeGroups;
  vector<Triangles> triangleGroups;
  vector<Tetrahedra> tetrahedraGroups;

  // Add ground
  auto [groundVertices, groundTriangles] = silk::makeBox();
  // groundVertices.array() *= 0.3;
  groundTriangles = silk::appendTriangles(vertexPositions, triangleGroups, groundVertices, groundTriangles);


  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;

  silk::registerInPolyscope(vertexPositions, pointGroups, edgeGroups, triangleGroups, tetrahedraGroups);

  polyscope::show();
}