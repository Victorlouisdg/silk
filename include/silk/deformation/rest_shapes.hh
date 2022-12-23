#include "silk/types.hh"

using namespace std;

namespace silk {

vector<Eigen::Matrix2d> makeRestShapesFromCurrentPositions(const VertexPositions &vertexPositions,
                                                           const Triangles &triangles) {
  vector<Eigen::Matrix2d> restShapes;

  for (int triangleIndex = 0; triangleIndex < triangles.rows(); ++triangleIndex) {
    Eigen::Vector3d x0 = vertexPositions.row(triangles(triangleIndex, 0));
    Eigen::Vector3d x1 = vertexPositions.row(triangles(triangleIndex, 1));
    Eigen::Vector3d x2 = vertexPositions.row(triangles(triangleIndex, 2));

    Eigen::Vector3d x1x0 = x1 - x0;
    Eigen::Vector3d x2x0 = x2 - x0;

    Eigen::Vector2d u1u0(x1x0.norm(), 0.0);

    double angle = acos(x1x0.dot(x2x0) / (x1x0.norm() * (x2x0).norm()));
    Eigen::Vector2d u2u0(cos(angle) * (x2x0).norm(), sin(angle) * (x2x0).norm());

    Eigen::Matrix2d restShape;
    restShape << u1u0, u2u0;  // It is important that these are column vectors.
    restShapes.push_back(restShape);
  }
  return restShapes;
}

}  // namespace silk