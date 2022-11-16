#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/surface_mesh.h"

#include <iostream>
#include <torch/torch.h>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

int main() {
  torch::Tensor tensor = torch::rand({2, 3});
  std::cout << tensor << std::endl;

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = readManifoldSurfaceMesh("spot.obj");

  for (Vertex v : mesh->vertices()) {
    std::cout << "Vertex " << v << " has degree " << v.degree() << "\n";
    for (Face fn : v.adjacentFaces()) {
      std::cout << "  incident on face " << fn << "\n";
    }
  }

  polyscope::init(); // initialize the gui

  // add the mesh to the gui
  auto *psMesh = polyscope::registerSurfaceMesh(
      "my mesh", geometry->vertexPositions, mesh->getFaceVertexList());

  VertexData<Vector2> field = computeSmoothestVertexDirectionField(*geometry);
  geometry->requireVertexTangentBasis();
  VertexData<Vector3> vBasisX(*mesh);
  for (Vertex v : mesh->vertices()) {
    vBasisX[v] = geometry->vertexTangentBasis[v][0];
  }
  psMesh->setVertexTangentBasisX(vBasisX);
  psMesh->addVertexIntrinsicVectorQuantity("vectors", field);

  IntrinsicGeometryInterface &geometry2 = *geometry;
  geometry2.requireFaceAreas();

  FaceData<double> areas(*mesh);
  int i = 0;
  for (Face f : mesh->faces()) {

    // Managed array holding quantity
    areas[i] = geometry2.faceAreas[f];
    i++;
  }
  psMesh->addFaceScalarQuantity("areas", areas);
  polyscope::show(); // pass control to the gui until the user exits
}