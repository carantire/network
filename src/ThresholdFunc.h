#pragma once

#include "utils.h"
#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>
#include <iostream>

namespace network {

enum class ThresholdId { Sigmoid, ReLu, LeakyRelu, Default };

struct ThresholdDatabase {

  template <ThresholdId> static double evaluate_0(double);

  template <ThresholdId> static Matrix evaluate_0_mat(const Matrix &);

  template <ThresholdId> static double evaluate_1(double);

  template <ThresholdId> static Matrix evaluate_1_mat(const Matrix &);

  template <> inline double evaluate_0<ThresholdId::Sigmoid>(double x) {

    double result = 1 / (1 + std::exp(-x));

    assert(std::isfinite(result));
    return result;
  }

  template <> inline double evaluate_1<ThresholdId::Sigmoid>(double x) {
    assert(std::isfinite(x));
    double result = 1. / (std::exp(-x) + std::exp(x) + 2.);
    assert(std::isfinite(result));
    return result;
  }

  template <> inline double evaluate_0<ThresholdId::ReLu>(double x) {
    return x * (x > 0);
  }

  template <> inline double evaluate_1<ThresholdId::ReLu>(double x) {
    return x > 0 ? 1 : 0;
  }
  template <> inline double evaluate_0<ThresholdId::LeakyRelu>(double x) {
    return x > 0 ? x : exp(-2) * x;
  }

  template <> inline double evaluate_1<ThresholdId::LeakyRelu>(double x) {
    return x > 0 ? 1 : exp(-2);
  }
  template <>
  inline Matrix evaluate_0_mat<ThresholdId::LeakyRelu>(const Matrix &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_0<ThresholdId::LeakyRelu>(x); });
  }
  template <>
  inline Matrix evaluate_1_mat<ThresholdId::LeakyRelu>(const Matrix &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_1<ThresholdId::LeakyRelu>(x); });
  }
  template <>
  inline Matrix evaluate_0_mat<ThresholdId::Default>(const Matrix &mat) {
    return mat;
  }

  template <>
  inline Matrix evaluate_1_mat<ThresholdId::Default>(const Matrix &mat) {
    return mat;
  }

  template <>
  inline Matrix evaluate_0_mat<ThresholdId::Sigmoid>(const Matrix &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_0<ThresholdId::Sigmoid>(x); });
  }

  template <>
  inline Matrix evaluate_1_mat<ThresholdId::Sigmoid>(const Matrix &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_1<ThresholdId::Sigmoid>(x); });
  }

  template <>
  inline Matrix evaluate_0_mat<ThresholdId::ReLu>(const Matrix &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_0<ThresholdId::ReLu>(x); });
  }

  template <>
  inline Matrix evaluate_1_mat<ThresholdId::ReLu>(const Matrix &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_1<ThresholdId::ReLu>(x); });
  }
};

class ThresholdFunc {
public:
  using FunctionType = std::function<Matrix(const Matrix &)>;

  ThresholdFunc(FunctionType evaluate_0, FunctionType evaluate_1);

  template <ThresholdId Id> static ThresholdFunc create() {
    return ThresholdFunc(ThresholdDatabase::evaluate_0_mat<Id>,
                         ThresholdDatabase::evaluate_1_mat<Id>);
  }

  static ThresholdFunc create(ThresholdId threshold);

  Matrix apply(const Matrix &layer_val) const;

  Matrix derive(const Matrix &layer_val) const;

  bool check_empty();

private:
  FunctionType evaluate_0_;
  FunctionType evaluate_1_;
};
} // namespace network
