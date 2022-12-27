#pragma once

#include <Eigen/Core>

/* Some simple aliases to keep function signatures readable. */
using VertexPositions = Eigen::Matrix<double, Eigen::Dynamic, 3>;
using Points = Eigen::ArrayXi;
using Edges = Eigen::ArrayX2i;
using Triangles = Eigen::ArrayX3i;
using Tetrahedra = Eigen::ArrayX4i;

template<typename ScalarT> using DeformationGradient = Eigen::Matrix<ScalarT, 3, 2>;
template<typename ScalarT> using ElementEnergyFunction = std::function<ScalarT(const DeformationGradient<ScalarT> &)>;

using EnergyDerivatives = std::tuple<double, Eigen::VectorXd, Eigen::SparseMatrix<double>>;
using EnergyFunction = std::function<EnergyDerivatives(const Eigen::VectorXd &)>;