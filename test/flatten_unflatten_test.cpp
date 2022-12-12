#include "silk/conversions.hh"
#include <Eigen/Core>
#include <gtest/gtest.h>

using namespace std;
using namespace geometrycentral;
using namespace geometrycentral::surface;

// Demonstrate some basic assertions.
TEST(UnflattenFlatten, BasicAssertions) {
  Eigen::VectorXd v = Eigen::VectorXd::LinSpaced(12, 1, 12);

  Eigen::MatrixXd m = silk::unflatten(v);
  Eigen::VectorXd v2 = silk::flatten(m);

  EXPECT_TRUE(v2.isApprox(v)) << "Unflatten followed by flatten should lead to the original vector.";
}