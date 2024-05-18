//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Layer.h"
#include "utils.h"
#include <iostream>

namespace network {

Layer::Layer(ThresholdId id, Index in_size, Index out_size, int seed,
             double normalize)
    : ThresholdFunc_(ThresholdFunc::create(id)),
      A_(getNormal(out_size, in_size, seed, normalize)),
      b_(getNormal(out_size, 1, seed, normalize)) {}

Matrix Layer::apply_linear(const Matrix &x) const {

  return ((A_ * x).colwise() + b_).eval();
}

Matrix Layer::apply_threshold(const Matrix &value) const {
  return ThresholdFunc_.apply(value);
}

Matrix Layer::derive(const Matrix &applied_values) const {
  return ThresholdFunc_.derive(applied_values);
}

Matrix Layer::derive_mat(const Matrix &applied_values_mat,
                         const Matrix &grad) const {
  return (derive(applied_values_mat).cwiseProduct(grad)).eval();
}

Matrix Layer::gradx(const Matrix &grad, const Matrix &applied_val) const {
  return (A_.transpose() * derive_mat(applied_val, grad)).eval();
}

void Layer::apply_gradA(const Matrix &values, const Matrix &grad,
                        const Matrix &applied_val, double step) {
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

void Layer::apply_gradb(const Matrix &grad, const Matrix &applied_val,
                        double step) {
  auto diff = (derive_mat(applied_val, grad) *
               Vector::Ones(applied_val.cols()) / applied_val.cols())
                  .eval();
  b_ -= step * diff;
}
RandGen &getUrng(int seed) {
  static RandGen urng = seed;
  return urng;
}

Matrix Layer::getNormal(Index rows, Index columns, int seed, double normalize) {
  assert(rows > 0 && "rows must be positive");
  assert(columns > 0 && "columns must be positive");
  return Eigen::Rand::normal<Matrix>(rows, columns, getUrng(seed)) * normalize;
}

Index Layer::Get_Input_Dim() const { return A_.cols(); };

Matrix Layer::Get_Mat() const { return A_; }

Layer Layer::ReadParams(std::ifstream &in) {
  ThresholdId Id;
  Index in_size, out_size;
  in.read(reinterpret_cast<char*>(&Id), sizeof(Id));
  in.read(reinterpret_cast<char*>(&in_size), sizeof(in_size));
  in.read(reinterpret_cast<char*>(&out_size), sizeof(out_size));
  Layer layer = Layer(Id, in_size, out_size, 1, 1.);
  in.read((char *)layer.A_.data(),
          in_size * out_size * sizeof(typename Matrix::Scalar));
  in.read((char *)layer.b_.data(), out_size * 1 * sizeof(typename Vector::Scalar));
  return layer;
}

void Layer::WriteParams(std::ofstream &out) const {
  typename Matrix::Index rows = A_.rows();
  typename Matrix::Index cols = A_.cols();
  ThresholdId Id = ThresholdFunc_.GetId();
  out.write(reinterpret_cast<char*>(&Id), sizeof(Id));
  out.write(reinterpret_cast<char*>(&cols), sizeof(cols));
  out.write(reinterpret_cast<char*>(&rows), sizeof(rows));
  out.write((char *)A_.data(), rows * cols * sizeof(typename Matrix::Scalar));
  out.write((char *)b_.data(), rows * 1 * sizeof(typename Vector ::Scalar));
}

} // namespace network
