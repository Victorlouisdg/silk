#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "silk/types.hh"
#include <iostream>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

namespace silk {

/**
 * @brief Extend the existing vertices with new ones and add new triangles.
 */
Triangles appendTriangles(VertexPositions &vertices,
                          vector<Triangles> &triangleGroups,
                          const VertexPositions &newVertices,
                          const Triangles &newTriangles) {
  int amountOfVerticesBeforeInsertion = vertices.rows();

  // Add the new vertices at the bottom.
  vertices.conservativeResize(vertices.rows() + newVertices.rows(), 3);
  vertices.bottomRows(newVertices.rows()) = newVertices;

  // Add the new triangles, but also offset them by the amount of rows that were already present.
  Triangles reindexedNewTriangles = newTriangles + amountOfVerticesBeforeInsertion;
  triangleGroups.push_back(reindexedNewTriangles);
  return reindexedNewTriangles;
}

/**
 * @brief Creates a right angle triangle with vertex 0 at the origin, vertex 1 at (0, 1, 0) and vertex 2 at (1,0,0).
 * With the vertices in this order, the triangle is oriented so that the normal points in the positive z direction.
 */
tuple<VertexPositions, Triangles> makeRightAngleTriangle() {
  Eigen::RowVector3d v0(0.0, 0.0, 0.0);
  Eigen::RowVector3d v1(1.0, 0.0, 0.0);
  Eigen::RowVector3d v2(0.0, 1.0, 0.0);

  VertexPositions vertexPositions(3, 3);
  vertexPositions << v0, v1, v2;

  Eigen::RowVector3i triangle{0, 1, 2};
  Triangles triangles(1, 3);
  triangles << triangle;

  return std::make_tuple(vertexPositions, triangles);
}

tuple<VertexPositions, Triangles> makeRegularTriangle() {
  // I want a triangle with the same area 1/2 as the right angle triangle above.
  // https://www.wolframalpha.com/input?i=length+of+a+regular+triangle+with+area+1%2F2
  double edgeLength = 1.07457;
  double height = 0.930605;
  Eigen::RowVector3d v0(0.0, 0.0, 0.0);
  Eigen::RowVector3d v1(edgeLength, 0.0, 0.0);
  Eigen::RowVector3d v2(edgeLength / 2.0, height, 0.0);

  VertexPositions vertexPositions(3, 3);
  vertexPositions << v0, v1, v2;

  Eigen::RowVector3i triangle{0, 1, 2};
  Triangles triangles(1, 3);
  triangles << triangle;

  return std::make_tuple(vertexPositions, triangles);
}

}  // namespace silk