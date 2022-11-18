#include <torch/torch.h>

#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"

using namespace geometrycentral;
using namespace geometrycentral::surface;

using namespace torch;
using namespace torch::indexing;

Tensor deformationGradient(Tensor positions, Tensor positions_uv) {
  // Read out the rows.
  Tensor u0 = positions_uv.index({0, Slice()});
  Tensor u1 = positions_uv.index({1, Slice()});
  Tensor u2 = positions_uv.index({2, Slice()});

  Tensor Dm = torch::column_stack({u1 - u0, u2 - u0});
  Tensor Dm_inv = torch::inverse(Dm);

  Tensor x0 = positions.index({0, Slice()});
  Tensor x1 = positions.index({1, Slice()});
  Tensor x2 = positions.index({2, Slice()});
  Tensor Ds = torch::column_stack({x1 - x0, x2 - x0});

  return torch::matmul(Ds, Dm_inv);
}

void visualize(std::vector<std::vector<size_t>> triangles, std::vector<Vector3> vertexCoordinates) {
  SimplePolygonMesh simpleMesh(triangles, vertexCoordinates);

  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geometry;
  std::tie(mesh, geometry) = makeManifoldSurfaceMeshAndGeometry(simpleMesh.polygons, simpleMesh.vertexCoordinates);

  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;
  auto *psMesh = polyscope::registerSurfaceMesh("my mesh", geometry->vertexPositions, mesh->getFaceVertexList());
  // addFaceNormals(*geometry, *mesh, *psMesh);
  polyscope::show();
}

std::vector<Vector3> torchToGeometryCentral(Tensor vectorQuantityTensor) {
  std::vector<Vector3> vectorQuantities;

  for (int i = 0; i < vectorQuantityTensor.sizes()[0]; i++) {
    Tensor quantityRow = vectorQuantityTensor.index({i, Slice()});
    Vector3 vectorQuantity{quantityRow.index({0}).item<double>(),
                           quantityRow.index({1}).item<double>(),
                           quantityRow.index({2}).item<double>()};
    vectorQuantities.push_back(vectorQuantity);
  }
  return vectorQuantities;
}

void visualizeTorch(std::vector<std::vector<size_t>> triangles,
                    Tensor positions,
                    std::vector<Tensor> customVertexData) {
  std::vector<Vector3> vertexCoordinates = torchToGeometryCentral(positions);
  visualize(triangles, vertexCoordinates);
}

Tensor energy(Tensor positions) {
  // Temp hardcorded UVs
  Tensor positions_orignal = torch::tensor({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.5, 0.0}});
  Tensor positions_uv = positions_orignal.index({Slice(), Slice(None, 2)});  // positions[:, :2];

  Tensor F = deformationGradient(positions, positions_uv);
  Tensor wu = F.index({Slice(), 0});
  Tensor wv = F.index({Slice(), 1});

  Tensor Cu = torch::norm(wu) - 1.0;
  Tensor Cv = torch::norm(wv) - 1.0;

  Tensor Eu = 0.5 * Cu * Cu;
  Tensor Ev = 0.5 * Cv * Cv;

  Tensor E = Eu + Ev;
  return E;
}

int main() {
  // Torch part
  Tensor positions_orignal = torch::tensor({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 0.5, 0.0}},
                                           torch::requires_grad());
  Tensor positions = torch::tensor({{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.5, 1.0, 0.0}}, torch::requires_grad());

  std::cout << positions << std::endl;

  Tensor positions_uv = positions_orignal.index({Slice(), Slice(None, 2)});  // positions[:, :2];
  std::cout << positions_uv << std::endl;

  Tensor F = deformationGradient(positions, positions_uv);
  Tensor wu = F.index({Slice(), 0});
  Tensor wv = F.index({Slice(), 1});

  Tensor Cu = torch::norm(wu) - 1.0;
  Tensor Cv = torch::norm(wv) - 1.0;

  Tensor Eu = 0.5 * Cu * Cu;
  Tensor Ev = 0.5 * Cv * Cv;

  Tensor E = Eu + Ev;

  // E.backward();
  // positions.grad();

  std::cout << wu << std::endl;
  std::cout << wv << std::endl;

  std::cout << Eu << std::endl;

  auto grad_output = torch::ones_like(E);
  auto gradient = torch::autograd::grad({E}, {positions}, {grad_output}, true, true)[0];

  std::cout << gradient << std::endl;

  // This is a higher order derivative because we take the gradient,
  // then reduce it to a scalar, and then can call backward() on it.
  // Followed by .grad() on the inputs. This .grad() is the gradient of a function of the first grad.
  auto gradient_penalty = torch::pow((gradient.norm(2, 1) - 1), 2).mean();
  std::cout << gradient_penalty << std::endl;

  gradient_penalty.backward(torch::ones_like(gradient_penalty), true);
  std::cout << positions.grad() << std::endl;

  gradient.index({0, 1}).backward();
  std::cout << positions.grad() << std::endl;  // small part of the hessian

  // Tensor position0 = positions.index({0, Slice()});
  // position0.set_requires_grad(true);
  // position0.retain_grad();  // Does not seem to work :/
  // auto grad0_output = torch::ones_like(E);
  // auto gradient0 = torch::autograd::grad({E}, {position0}, {grad0_output}, true, true)[0];
  // std::cout << gradient0 << std::endl;

  // auto hessian = torch::autograd::functional::hessian(energy, )

  std::vector<size_t> triangle{0, 1, 2};
  std::vector<std::vector<size_t>> triangles{triangle};
  visualizeTorch(triangles, positions, {});
}