//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Layer.h"
#include <iostream>

namespace network {
using Index = Layer::Index;
using VectorXd = Layer::VectorXd;
using MatrixXd = Layer::MatrixXd;
using RandGen = Eigen::Rand::Vmt19937_64;

Layer::Layer(ThresholdId id, Index in_size, Index out_size)
    : ThresholdFunc_(ThresholdFunc::create(id)),
      A_(getNormal(out_size, in_size)), b_(getNormal(out_size, 1)) {}

MatrixXd Layer::apply_linear(const MatrixXd &x) const {
  return (A_ * x + b_ * VectorXd::Ones(x.cols()).transpose()).eval();
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
    assert(std::isfinite(applied_values_mat.col(i).array().maxCoeff()) &&
           "input is inf");
    res.col(i) = derive(applied_values_mat.col(i)) * grad.col(i);
    assert(res.col(i).norm() > 0);
  }
  return res.eval();
}

MatrixXd Layer::gradx(const MatrixXd &grad, const MatrixXd &applied_val) const {
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

  A_ += step * diff;
}

void Layer::apply_gradb(const MatrixXd &grad, const MatrixXd &applied_val,
                        double step) {
  auto diff = (derive_mat(applied_val, grad) *
               VectorXd::Ones(applied_val.cols()) / applied_val.cols())
                  .eval();
  b_ += step * diff;
}

RandGen &getUrng() {
  static RandGen urng = 1;
  return urng;
}

MatrixXd Layer::getNormal(Index rows, Index columns) {
  assert(rows > 0 && "rows must be positive");
  assert(columns > 0 && "columns must be positive");
  MatrixXd A(rows, columns);
  static std::mt19937 rng(std::random_device{}());
  static std::normal_distribution<> nd(0.0, sqrt(2.0 / (columns)));

  A = A.unaryExpr([](double dummy) { return nd(rng); });
  return A;
}

Index Layer::Get_Input_Dim() const { return A_.cols(); };

void Layer::Print_Mat() const {
  std::cout << A_.leftCols(5).transpose().leftCols(5).transpose() << "\n\n\n";
}

} // namespace network
