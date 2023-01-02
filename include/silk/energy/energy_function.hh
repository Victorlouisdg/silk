#pragma once

#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include <iostream>

#include <TinyAD/ScalarFunction.hh>

#include "silk/types.hh"
#include <ipc/ipc.hpp>

using namespace std;

namespace silk {

/**
 * @brief This class standardizes the interface to TinyAD ScalarFunctions and the IPC-toolkit energy functions.
 * I've chosen to follow the TinyAD ScalarFunction class.
 *
 */
class Energy {
 public:
  virtual double eval(const Eigen::VectorXd &x) = 0;
  // virtual double operator()(const Eigen::VectorXd &x) const = 0;

  double operator()(const Eigen::VectorXd &x) {
    return this->eval(x);
  }

  virtual tuple<double, Eigen::VectorXd> eval_with_gradient(const Eigen::VectorXd &x) = 0;
  virtual tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_derivatives(
      const Eigen::VectorXd &x) = 0;
  virtual tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_hessian_proj(
      const Eigen::VectorXd &x) = 0;
};

class TinyADEnergy : public Energy {
  TinyAD::ScalarFunction<3, double, Eigen::Index> scalarFunction;

 public:
  TinyADEnergy(TinyAD::ScalarFunction<3, double, Eigen::Index> &scalarFunction) {
    // I don't fully understand of the consequences of this move. I should document this once I do.
    this->scalarFunction = std::move(scalarFunction);
  };

  double eval(const Eigen::VectorXd &x) override {
    return scalarFunction.eval(x);
  }

  // double operator()(const Eigen::VectorXd &x) override {
  //   return this->eval(x);
  // }

  tuple<double, Eigen::VectorXd> eval_with_gradient(const Eigen::VectorXd &x) override {
    return scalarFunction.eval_with_gradient(x);
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_derivatives(
      const Eigen::VectorXd &x) override {
    return scalarFunction.eval_with_derivatives(x);
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_hessian_proj(
      const Eigen::VectorXd &x) override {
    return scalarFunction.eval_with_hessian_proj(x);
  }
};

class IPCBarrierEnergy : public Energy {
  ipc::CollisionMesh collisionMesh;
  ipc::Constraints constraintSet;
  ipc::BroadPhaseMethod method = ipc::BroadPhaseMethod::HASH_GRID;
  double dhat;

 public:
  IPCBarrierEnergy(ipc::CollisionMesh &collisionMesh, ipc::Constraints &constraintSet, double dhat) {
    this->collisionMesh = collisionMesh;
    this->constraintSet = constraintSet;
    this->dhat = dhat;
  }

  double eval(const Eigen::VectorXd &x) override {
    Eigen::MatrixXd collisionV = collisionMesh.vertices(silk::unflatten(x));
    this->constraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);
    double barrierPotential = ipc::compute_barrier_potential(collisionMesh, collisionV, constraintSet, dhat);
    return barrierPotential;
  }

  tuple<double, Eigen::VectorXd> eval_with_gradient(const Eigen::VectorXd &x) override {
    /* Warning: currently requires eval to be called first. */
    Eigen::MatrixXd collisionV = collisionMesh.vertices(silk::unflatten(x));
    this->constraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);
    double barrierPotential = ipc::compute_barrier_potential(collisionMesh, collisionV, constraintSet, dhat);
    Eigen::VectorXd barrierGradient = ipc::compute_barrier_potential_gradient(
        collisionMesh, collisionV, constraintSet, dhat);
    return {barrierPotential, barrierGradient};
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_derivatives(
      const Eigen::VectorXd &x) override {
    Eigen::MatrixXd collisionV = collisionMesh.vertices(silk::unflatten(x));
    this->constraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);
    double barrierPotential = ipc::compute_barrier_potential(collisionMesh, collisionV, constraintSet, dhat);
    Eigen::VectorXd barrierGradient = ipc::compute_barrier_potential_gradient(
        collisionMesh, collisionV, constraintSet, dhat);
    Eigen::SparseMatrix<double> barrierHessian = ipc::compute_barrier_potential_hessian(
        collisionMesh, collisionV, constraintSet, dhat, false);
    return {barrierPotential, barrierGradient, barrierHessian};
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_hessian_proj(
      const Eigen::VectorXd &x) override {
    /* Warning: currently requires eval to be called first. */

    Eigen::MatrixXd collisionV = collisionMesh.vertices(silk::unflatten(x));

    this->constraintSet.build(collisionMesh, collisionV, dhat, /*dmin=*/0, method);
    double barrierPotential = ipc::compute_barrier_potential(collisionMesh, collisionV, constraintSet, dhat);
    Eigen::VectorXd barrierGradient = ipc::compute_barrier_potential_gradient(
        collisionMesh, collisionV, constraintSet, dhat);
    Eigen::SparseMatrix<double> barrierHessianProj = ipc::compute_barrier_potential_hessian(
        collisionMesh, collisionV, constraintSet, dhat);
    return {barrierPotential, barrierGradient, barrierHessianProj};
  }
};

/**
 * @brief Simple convenience class that takes a map of energies and weights and sums them together.
 */
class AdditiveEnergy : public Energy {
 public:
  map<string, Energy *> energies;
  map<string, double> weights;

  AdditiveEnergy(map<string, Energy *> energies, map<string, double> weights) {
    this->energies = energies;
    this->weights = weights;
  }

  double eval(const Eigen::VectorXd &x) override {
    double sum = 0.0;
    for (auto const &[name, energy_ptr] : energies) {
      double weight = weights.at(name);
      sum += weight * energy_ptr->eval(x);
    }
    return sum;
  }

  tuple<double, Eigen::VectorXd> eval_with_gradient(const Eigen::VectorXd &x) override {
    double sum = 0.0;
    Eigen::VectorXd summedGradient = Eigen::VectorXd::Zero(x.size());
    for (auto const &[name, energy_ptr] : energies) {
      double weight = weights.at(name);
      auto [energy, energyGradient] = energy_ptr->eval_with_gradient(x);
      sum += weight * energy;
      summedGradient += weight * energyGradient;
    }
    return {sum, summedGradient};
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_derivatives(
      const Eigen::VectorXd &x) override {
    double sum = 0.0;
    Eigen::VectorXd summedGradient = Eigen::VectorXd::Zero(x.size());
    Eigen::SparseMatrix<double> summedHessian = Eigen::SparseMatrix<double>(x.size(), x.size());
    for (auto const &[name, energy_ptr] : energies) {
      double weight = weights.at(name);
      auto [energy, energyGradient, energyHessian] = energy_ptr->eval_with_derivatives(x);
      sum += weight * energy;
      summedGradient += weight * energyGradient;
      summedHessian += weight * energyHessian;
    }
    return {sum, summedGradient, summedHessian};
  }

  tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>> eval_with_hessian_proj(
      const Eigen::VectorXd &x) override {
    double sum = 0.0;
    Eigen::VectorXd summedGradient = Eigen::VectorXd::Zero(x.size());
    Eigen::SparseMatrix<double> summedHessian = Eigen::SparseMatrix<double>(x.size(), x.size());
    for (auto const &[name, energy_ptr] : energies) {
      double weight = weights.at(name);
      auto [energy, energyGradient, energyHessian] = energy_ptr->eval_with_hessian_proj(x);
      sum += weight * energy;
      summedGradient += weight * energyGradient;
      summedHessian += weight * energyHessian;
    }
    return {sum, summedGradient, summedHessian};
  }
};

}  // namespace silk