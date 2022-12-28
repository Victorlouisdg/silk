#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <iostream>

#include <TinyAD/ScalarFunction.hh>

#include "silk/types.hh"

using namespace std;

namespace silk {

/**
 * @brief This class standardizes the interface to TinyAD ScalarFunctions and the IPC-toolkit energy functions.
 * I've chosen to follow the TinyAD ScalarFunction class.
 *
 */
class Energy {
 public:
  virtual double eval(const Eigen::VectorXd &x) const = 0;
  virtual double operator()(const Eigen::VectorXd &x) const = 0;
  virtual tuple<double, Eigen::VectorXd> eval_with_gradient(const Eigen::VectorXd &x) const = 0;
  virtual tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_derivatives(
      const Eigen::VectorXd &x) const = 0;
  virtual tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_hessian_proj(
      const Eigen::VectorXd &x) const = 0;
};

class TinyADEnergy : public Energy {
  TinyAD::ScalarFunction<3, double, Eigen::Index> scalarFunction;

 public:
  TinyADEnergy(TinyAD::ScalarFunction<3, double, Eigen::Index> &scalarFunction) {
    // I don't fully understand of the consequences of this move. I should document this once I do.
    this->scalarFunction = std::move(scalarFunction);
  };

  double eval(const Eigen::VectorXd &x) const override {
    return scalarFunction.eval(x);
  }

  double operator()(const Eigen::VectorXd &x) const override {
    return this->eval(x);
  }

  tuple<double, Eigen::VectorXd> eval_with_gradient(const Eigen::VectorXd &x) const override {
    return scalarFunction.eval_with_gradient(x);
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_derivatives(
      const Eigen::VectorXd &x) const override {
    return scalarFunction.eval_with_derivatives(x);
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_hessian_proj(
      const Eigen::VectorXd &x) const override {
    return scalarFunction.eval_with_hessian_proj(x);
  }
};

}  // namespace silk