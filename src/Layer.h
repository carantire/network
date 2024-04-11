#pragma once

#include "ThresholdFunc.h"
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

class Layer {
public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  using Index = Eigen::Index;

  Layer(ThresholdId id, Index in_size, Index out_size, int seed, double normalize);

  MatrixXd apply_linear(const MatrixXd &values) const;

  MatrixXd apply_threshold(const MatrixXd &value) const;

  MatrixXd derive(const MatrixXd &applied_values) const;

  MatrixXd derive_mat(const MatrixXd &applied_values,
                      const MatrixXd &grad) const;

  MatrixXd gradx(const MatrixXd &grad, const MatrixXd &applied_values) const;

  void apply_gradA(const MatrixXd &values, const MatrixXd &grad,
                   const MatrixXd &applied_values, double step);

  void apply_gradb(const MatrixXd &grad, const MatrixXd &applied_values,
                   double step);

  Index Get_Input_Dim() const;

  MatrixXd Get_Mat() const;

private:
  static MatrixXd getNormal(Index rows, Index columns, int seed, double normalize);

  ThresholdFunc ThresholdFunc_;

  MatrixXd A_;
  VectorXd b_;
};
} // namespace network
