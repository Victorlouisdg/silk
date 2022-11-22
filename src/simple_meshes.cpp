#include <iostream>
#include <silk/simple_meshes.hh>

namespace silk {
tuple<unique_ptr<ManifoldSurfaceMesh>, unique_ptr<VertexPositionGeometry>> makeTwoTriangleSquare() {
  // Setup mesh and rest shape
  vector<size_t> triangle0{0, 1, 2};
  vector<size_t> triangle1{0, 2, 3};
  vector<vector<size_t>> triangles{triangle0, triangle1};

  Vector3 v0{0.0, 0.0, 0.0};
  Vector3 v1{1.0, 0.0, 0.0};
  Vector3 v2{1.0, 1.0, 0.0};
  Vector3 v3{0.0, 1.0, 0.0};

  vector<Vector3> vertexCoordinates{v0, v1, v2, v3};
  SimplePolygonMesh simpleMesh(triangles, vertexCoordinates);
  return makeManifoldSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);
};
}  // namespace silk
