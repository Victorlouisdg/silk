#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/meshio.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"

#include <iostream>

#include "polyscope/curve_network.h"
#include "polyscope/polyscope.h"
#include "polyscope/surface_mesh.h"
#include "polyscope/volume_mesh.h"

#include "silk/simple_meshes.hh"
#include "silk/visualization.hh"

#include <TinyAD/ScalarFunction.hh>
#include <igl/triangle/triangulate.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

int main() {
  polyscope::init();
  polyscope::view::upDir = polyscope::UpDir::ZUp;

  Eigen::MatrixXd V(2, 3);
  Eigen::RowVector3d v0(0, 0, 0);
  Eigen::RowVector3d v1(1, 0, 0);
  V << v0, v1;

  Eigen::MatrixXi E(1, 2);
  E << 0, 1;

  Eigen::VectorXd restLengths(E.rows());
  restLengths << (v0 - v1).norm();
  restLengths(0) = 1.1;

  auto elasticPotentialFunction = TinyAD::scalar_function<3>(TinyAD::range(V.rows()));

  elasticPotentialFunction.add_elements<2>(TinyAD::range(E.rows()), [&](auto &element) -> TINYAD_SCALAR_TYPE(element) {
    // Evaluate element using either double or TinyAD::Double
    using ScalarT = TINYAD_SCALAR_TYPE(element);
    int idx = element.handle;
    Eigen::Vector3<ScalarT> x0 = element.variables(E(idx, 0));
    Eigen::Vector3<ScalarT> x1 = element.variables(E(idx, 1));

    Eigen::Vector3<ScalarT> deltaX = (x0 - x1);
    ScalarT length = deltaX.norm();
    ScalarT restLength = restLengths(idx);
    ScalarT elongation = length - restLength;
    ScalarT springEnergy = 0.5 * elongation * elongation;

    return springEnergy;
  });

  Eigen::VectorXd x0 = elasticPotentialFunction.x_from_data([&](int v_idx) { return V.row(v_idx); });
  auto [elastic, elasticGradient] = elasticPotentialFunction.eval_with_gradient(x0);

  std::cout << " e: " << elastic << std::endl;
  std::cout << " g: " << elasticGradient.transpose() << std::endl;

  Eigen::MatrixXd gradient(elasticGradient.rows() / 3, 3);
  elasticPotentialFunction.x_to_data(
      elasticGradient, [&](int v_idx, const Eigen::Vector3d &vectorData) { gradient.row(v_idx) = vectorData; });

  polyscope::registerCurveNetwork("my network", V, E);
  polyscope::getCurveNetwork("my network")->addNodeVectorQuantity("forces", -gradient);

  // Show the GUI
  polyscope::show();
}