#pragma once
#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
namespace network {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;
using RandGen = Eigen::Rand::Vmt19937_64;
} // namespace network