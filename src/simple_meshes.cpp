#include <igl/triangle/triangulate.h>
#include <iostream>
#include <silk/simple_meshes.hh>

namespace silk {

Eigen::Vector3d make3D(Eigen::Vector2d v2) {
  Eigen::Vector3d v3;
  v3 << v2, 0.0;
  return v3;
}

tuple<unique_ptr<ManifoldSurfaceMesh>, unique_ptr<VertexPositionGeometry>> makeSingleTriangle() {
  // Setup mesh and rest shape
  vector<size_t> triangle0{0, 1, 2};
  vector<vector<size_t>> triangles{triangle0};

  Vector3 v0{0.0, 0.0, 0.0};
  Vector3 v1{1.0, 0.0, 0.0};
  Vector3 v2{0.0, 1.0, 0.0};

  vector<Vector3> vertexCoordinates{v0, v1, v2};
  SimplePolygonMesh simpleMesh(triangles, vertexCoordinates);
  return makeManifoldSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);
};

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

tuple<unique_ptr<ManifoldSurfaceMesh>, unique_ptr<VertexPositionGeometry>> makeSquare() {
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(4, 3);  // 3D vertex positions
  V << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0;

  // Eigen::MatrixXi F(1, 4);  // Mesh faces
  // F << 0, 1, 2, 3;

  Eigen::MatrixXi E(4, 2);  // Mesh edges
  E << 0, 1, 1, 2, 2, 3, 3, 0;

  Eigen::MatrixXd P = V.leftCols(2);  // 2D positions

  // Triangulated interior
  Eigen::MatrixXd V2;
  Eigen::MatrixXi F2;

  Eigen::MatrixXd H;

  igl::triangle::triangulate(P, E, H, "a0.005q", V2, F2);

  Eigen::MatrixXd Vnew = Eigen::MatrixXd::Zero(V2.rows(), 3);
  Vnew.leftCols(2) = V2;

  vector<Vector3> vertexCoordinates;

  for (int i = 0; i < V2.rows(); i++) {
    Eigen::Vector3d Vi = Vnew.row(i);
    vertexCoordinates.push_back(Vector3({Vi[0], Vi[1], Vi[2]}));
  }

  vector<vector<size_t>> triangles;

  for (int i = 0; i < F2.rows(); i++) {
    Eigen::Vector3i Fi = F2.row(i);
    vector<size_t> indices{(size_t)Fi[0], (size_t)Fi[1], (size_t)Fi[2]};
    triangles.push_back(indices);
  }

  SimplePolygonMesh simpleMesh(triangles, vertexCoordinates);
  return makeManifoldSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);
};

VertexData<Eigen::Vector2d> makeProjectedRestPositions(ManifoldSurfaceMesh &mesh, VertexPositionGeometry &geometry) {
  geometry.requireVertexPositions();
  VertexData<Eigen::Vector2d> vertexRestPositions(mesh);
  for (Vertex v : mesh.vertices()) {
    Vector3 position = geometry.vertexPositions[v];
    vertexRestPositions[v] = Eigen::Vector2d({position.x, position.y});
  }
  return vertexRestPositions;
}

tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeRegularTetrahedron() {
  Eigen::MatrixXd V = Eigen::MatrixXd::Zero(4, 3);  // 3D vertex positions

  // Adapted from:  https://www.danielsieger.com/blog/2021/01/03/generating-platonic-solids.html
  // choose coordinates on the unit sphere
  double a = 1.0 / 3.0;
  double b = sqrt(8.0 / 9.0);
  double c = sqrt(2.0 / 9.0);
  double d = sqrt(2.0 / 3.0);

  Eigen::RowVector3d v0(0, 0, 1);
  Eigen::RowVector3d v2(-c, d, -a);
  Eigen::RowVector3d v1(-c, -d, -a);
  Eigen::RowVector3d v3(b, 0, -a);
  V << v0, v1, v2, v3;

  Eigen::MatrixXi T(1, 4);  // tetrahedra
  T << 0, 1, 2, 3;

  return std::make_tuple(V, T);
};

tuple<Eigen::MatrixXd, Eigen::MatrixXi> makeStackedTetrahedra() {
  Eigen::MatrixXd V0;
  Eigen::MatrixXi T0;
  // Eigen::Matrix<int, Eigen::Dynamic, 4> T0;  // Test whether this results in efficiency gains
  std::tie(V0, T0) = silk::makeRegularTetrahedron();

  Eigen::MatrixXd V1;
  Eigen::MatrixXi T1;
  std::tie(V1, T1) = silk::makeRegularTetrahedron();
  V1.rightCols(1).array() += 1.5;  // Move tetrahedron up

  // Combine arrays of the tets for joint processing.
  Eigen::MatrixXd V = Eigen::MatrixXd(V0.rows() + V1.rows(), V0.cols());
  V.topRows(V0.rows()) = V0;
  V.bottomRows(V1.rows()) = V1;

  Eigen::MatrixXi T = Eigen::MatrixXi(T0.rows() + T1.rows(), T0.cols());
  T.topRows(T0.rows()) = T0;
  // Account for the offset the vertices of the second tetrahedron by the amount of vertices in the first.
  T.bottomRows(T1.rows()) = T1.array() + V0.rows();

  return std::make_tuple(V, T);
};

}  // namespace silk
