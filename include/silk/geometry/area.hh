#include "silk/types.hh"

using namespace std;

namespace silk {

/**
 * @brief Compute the area of a triangle given the positions of its three vertices.
 */
double triangleArea(Eigen::Vector3d const &x0, Eigen::Vector3d const &x1, Eigen::Vector3d const &x2) {
  return 0.5 * ((x1 - x0).cross(x2 - x0)).norm();
}

Eigen::VectorXd triangleAreas(const VertexPositions &vertexPositions, const Triangles &triangles) {
  Eigen::VectorXd areas(triangles.rows());

  for (int triangleIndex = 0; triangleIndex < triangles.rows(); ++triangleIndex) {
    Eigen::Vector3d x0 = vertexPositions.row(triangles(triangleIndex, 0));
    Eigen::Vector3d x1 = vertexPositions.row(triangles(triangleIndex, 1));
    Eigen::Vector3d x2 = vertexPositions.row(triangles(triangleIndex, 2));
    areas(triangleIndex) = triangleArea(x0, x1, x2);
  };
  return areas;
}

}  // namespace silk