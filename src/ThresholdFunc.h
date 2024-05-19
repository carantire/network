#pragma once

#include "utils.h"
#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>

namespace network {

enum class ThresholdId { Sigmoid, ReLu, LeakyRelu, Default };

struct ThresholdDatabase {

  template <ThresholdId> static Matrix evaluate_0(const Matrix &);

  template <ThresholdId> static Matrix evaluate_1(const Matrix &);

  template <> Matrix evaluate_0<ThresholdId::ReLu>(const Matrix &mat) {
    return (mat.array() > 0).select(mat, 0.0);
  }

  template <> Matrix evaluate_1<ThresholdId::ReLu>(const Matrix &mat) {
    return (mat.array() > 0).select(1.0 , Matrix::Zero(mat.rows(), mat.cols()));
  }

  template <>
  inline Matrix evaluate_0<ThresholdId::LeakyRelu>(const Matrix &mat) {
    return (mat.array() > 0).select(mat, mat*exp(-2));
  }
  template <> Matrix evaluate_1<ThresholdId::LeakyRelu>(const Matrix &mat) {
    return (mat.array() > 0).select(Matrix::Ones(mat.rows(), mat.cols()), exp(-2));
  }
  template <> Matrix evaluate_0<ThresholdId::Default>(const Matrix &mat) {
    return mat;
  }

  template <> Matrix evaluate_1<ThresholdId::Default>(const Matrix &mat) {
    return Matrix::Identity(mat.rows(), mat.cols());
  }

  template <> Matrix evaluate_0<ThresholdId::Sigmoid>(const Matrix &mat) {
    return ((-mat).array().exp() + 1.0).inverse();
  }

  template <> Matrix evaluate_1<ThresholdId::Sigmoid>(const Matrix &mat) {
    return (mat.array().exp() + (-mat).array().exp() + 2.0).inverse();
  }


};

class ThresholdFunc {
public:
  using FunctionType = std::function<Matrix(const Matrix &)>;

  ThresholdFunc(FunctionType evaluate_0, FunctionType evaluate_1,
                ThresholdId Id);

  template <ThresholdId Id> static ThresholdFunc create() {
    return ThresholdFunc(ThresholdDatabase::evaluate_0<Id>,
                         ThresholdDatabase::evaluate_1<Id>, Id);
  }

  static ThresholdFunc create(ThresholdId threshold);

  Matrix apply(const Matrix &layer_val) const;

  Matrix derive(const Matrix &layer_val) const;

  ThresholdId GetId() const;

  bool check_empty();

private:
  FunctionType evaluate_0_;
  FunctionType evaluate_1_;
  ThresholdId Id_;
};
} // namespace network
