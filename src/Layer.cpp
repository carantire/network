//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Layer.h"

namespace network {
using Index = Layer::Index;
using VectorXd = Layer::VectorXd;
using MatrixXd = Layer::MatrixXd;
using RandGen = Eigen::Rand::Vmt19937_64;

Layer::Layer(ThresholdId id, Index in_size, Index out_size)
    : ThresholdFunc_(ThresholdFunc::create(id)),
      A_(getNormal(out_size, in_size)), b_(getNormal(out_size, 1)) {}

MatrixXd Layer::apply_linear(const MatrixXd &x) const {
  return A_ * x + b_ * VectorXd::Ones(x.cols()).transpose();
}

MatrixXd Layer::apply_threshold(const MatrixXd &value) const {
  return ThresholdFunc_.apply(value);
}

MatrixXd Layer::derive(const VectorXd &vec) const {
  return ThresholdFunc_.derive(vec).asDiagonal();
}

MatrixXd Layer::derive_mat(const MatrixXd &applied_values_mat,
                           const MatrixXd &grad) const {
  MatrixXd res(applied_values_mat.rows(), applied_values_mat.cols());
  for (Index i = 0; i < res.cols(); ++i) {
    res.col(i) = derive(applied_values_mat.col(i)) * grad.col(i);
  }
  return res;
}

MatrixXd Layer::gradx(const MatrixXd &grad, const MatrixXd &applied_val) const {
  return A_.transpose() * derive_mat(applied_val, grad);
}

void Layer::apply_gradA(const MatrixXd &values, const MatrixXd &grad,
                        const MatrixXd &applied_val, double step) {
  auto diff =
      derive_mat(applied_val, grad) * values.transpose() / applied_val.cols();
  A_ -= step * diff;
}

void Layer::apply_gradb(const MatrixXd &grad, const MatrixXd &applied_val,
                        double step) {
  auto diff = derive_mat(applied_val, grad) *
              VectorXd::Ones(applied_val.cols()) / applied_val.cols();
  b_ -= step * diff;
}

RandGen &getUrng() {
  static RandGen urng = 1;
  return urng;
}

MatrixXd Layer::getNormal(Index rows, Index columns) {
  assert(rows > 0 && "rows must be positive");
  assert(columns > 0 && "columns must be positive");
  return Eigen::Rand::normal<MatrixXd>(rows, columns, getUrng());
}

Index Layer::Get_Input_Dim() const { return A_.cols(); };

} // namespace network
