//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Layer.h"
#include <iostream>

namespace network {


Layer::Layer(ThresholdId id, Index in_size, Index out_size, int seed, double normalize)
    : ThresholdFunc_(ThresholdFunc::create(id)),
      A_(getNormal(out_size, in_size, seed, normalize)), b_(getNormal(out_size, 1, seed, normalize)) {
}

Layer::MatrixXd Layer::apply_linear(const MatrixXd &x) const {

  return (A_ * x + b_ * VectorXd::Ones(x.cols()).transpose()).eval();
}

Layer::MatrixXd Layer::apply_threshold(const MatrixXd &value) const {
  return ThresholdFunc_.apply(value);
}

Layer::MatrixXd Layer::derive(const MatrixXd &applied_values) const {
  return ThresholdFunc_.derive(applied_values);
}

Layer::MatrixXd Layer::derive_mat(const MatrixXd &applied_values_mat,
                           const MatrixXd &grad) const {
  return (derive(applied_values_mat).cwiseProduct(grad)).eval();
}

Layer::MatrixXd Layer::gradx(const MatrixXd &grad, const MatrixXd &applied_val) const {
  return (A_.transpose() * derive_mat(applied_val, grad)).eval();
}

void Layer::apply_gradA(const MatrixXd &values, const MatrixXd &grad,
                        const MatrixXd &applied_val, double step) {
  assert(grad.cols() == applied_val.cols() &&
         "applied_values and gradient must have same cols");
  assert(grad.rows() == applied_val.rows() &&
         "applied_values and gradient must have same rows");
  assert(values.cols() == grad.cols() && "all matrices must have same cols");

  auto diff = (derive_mat(applied_val, grad) *
               (values.transpose() / applied_val.cols()).eval())
                  .eval();

  assert(diff.cols() == A_.cols());
  assert(diff.rows() == A_.rows());

  A_ -= step * diff;
}

void Layer::apply_gradb(const MatrixXd &grad, const MatrixXd &applied_val,
                        double step) {
  auto diff = (derive_mat(applied_val, grad) *
               VectorXd::Ones(applied_val.cols()) / applied_val.cols())
                  .eval();
  b_ -= step * diff;
}

Layer::RandGen &getUrng(int seed) {
  static Layer::RandGen urng = seed;
  return urng;
}

Layer::MatrixXd Layer::getNormal(Index rows, Index columns, int seed, double normalize) {
  assert(rows > 0 && "rows must be positive");
  assert(columns > 0 && "columns must be positive");
  return Eigen::Rand::normal<MatrixXd>(rows, columns, getUrng(seed)) * normalize;
}

Layer::Index Layer::Get_Input_Dim() const { return A_.cols(); };

Layer::MatrixXd Layer::Get_Mat() const { return A_; }

} // namespace network
