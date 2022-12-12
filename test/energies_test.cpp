#include <TinyAD/Support/GeometryCentral.hh>

#include <TinyAD/Scalar.hh>
#include <TinyAD/ScalarFunction.hh>

#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

#include "silk/conversions.hh"

#include "silk/energies.hh"
#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <gtest/gtest.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

using namespace silk;

// Demonstrate some basic assertions.
TEST(Energies, ZeroRestEnergy) {
  unique_ptr<ManifoldSurfaceMesh> mesh_pointer;
  unique_ptr<VertexPositionGeometry> geometry_pointer;
  std::tie(mesh_pointer, geometry_pointer) = silk::makeSingleTriangle();
  ManifoldSurfaceMesh &mesh = *mesh_pointer;
  VertexPositionGeometry &geometry = *geometry_pointer;

  geometry.requireVertexPositions();
  VertexData<Eigen::Vector2d> vertexRestPositions = silk::makeProjectedRestPositions(mesh, geometry);

  // Set up a function with 3D vertex positions as variables
  auto meshTriangleEnergies = TinyAD::scalar_function<3, double, VertexSet>(mesh.vertices());

  // Add objective term per face. Each connecting 3 vertices.
  meshTriangleEnergies.add_elements<3>(mesh.faces(), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    // Evaluate element using either double or TinyAD::Double
    using T = TINYAD_SCALAR_TYPE(element);

    // Get variable 2D vertex positions
    Face f = element.handle;
    Vertex v0 = f.halfedge().vertex();
    Vertex v1 = f.halfedge().next().vertex();
    Vertex v2 = f.halfedge().next().next().vertex();
    Eigen::Vector3<T> x0 = element.variables(v0);
    Eigen::Vector3<T> x1 = element.variables(v1);
    Eigen::Vector3<T> x2 = element.variables(v2);

    Eigen::Vector2d u0 = vertexRestPositions[v0];
    Eigen::Vector2d u1 = vertexRestPositions[v1];
    Eigen::Vector2d u2 = vertexRestPositions[v2];

    Eigen::Matrix<T, 3, 2> F = silk::deformationGradient(x0, x1, x2, u0, u1, u2);
    T stretchPotential = silk::baraffWitkinStretchPotential(F);
    T shearPotential = silk::baraffWitkinShearPotential(F);

    Eigen::Vector3d restNormal = silk::make3D(u1 - u0).cross(silk::make3D(u2 - u0));
    double restArea = restNormal.norm() / 2.0;

    // double stretchStiffness = 10000.0;
    double stretchStiffness = 100.0;
    double shearStiffness = 1.0;

    double areaFactor = sqrt(restArea);
    T shearEnergy = shearStiffness * areaFactor * shearPotential;
    T stretchEnergy = stretchStiffness * areaFactor * stretchPotential;
    T energy = stretchEnergy + shearEnergy;

    std::cout << "Stretch " << stretchPotential << std::endl;
    std::cout << "Shear " << shearPotential << std::endl;
    // This sadly doesn't work because T is seen a double sometimes.
    // std::cout << "Shear grad" << shearPotential.grad << std::endl;

    EXPECT_DOUBLE_EQ(TinyAD::to_passive(stretchPotential), 0.0)
        << "Stretch potential should be zero for a triangle in rest.";
    EXPECT_DOUBLE_EQ(TinyAD::to_passive(shearPotential), 0.0)
        << "Shear potential should be zero for a triangle in rest.";

    return energy;
  });

  Eigen::VectorXd positions = meshTriangleEnergies.x_from_data(
      [&](Vertex v) { return to_eigen(geometry.vertexPositions[v]); });

  auto [energy, gradient] = meshTriangleEnergies.eval_with_gradient(positions);

  std::cout << "Gradient: " << gradient << std::endl;
}