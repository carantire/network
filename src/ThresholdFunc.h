#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>
#include <iostream>

namespace network {

enum class ThresholdId { Sigmoid, ReLu, SoftMax, Default };

struct ThresholdDatabase {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  using Index = Eigen::Index;

  template <ThresholdId> static double evaluate_0(double);

  template <ThresholdId> static MatrixXd evaluate_0_mat(const MatrixXd &);

  template <ThresholdId> static double evaluate_1(double);

  template <ThresholdId> static MatrixXd evaluate_1_mat(const MatrixXd &);

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
  template <>
  inline MatrixXd evaluate_0_mat<ThresholdId::Default>(const MatrixXd &mat) {
    return mat;
  }

  template <>
  inline MatrixXd evaluate_1_mat<ThresholdId::Default>(const MatrixXd &mat) {
    return mat;
  }

  template <>
  inline MatrixXd evaluate_0_mat<ThresholdId::Sigmoid>(const MatrixXd &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_0<ThresholdId::Sigmoid>(x); });
  }

  template <>
  inline MatrixXd evaluate_1_mat<ThresholdId::Sigmoid>(const MatrixXd &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_1<ThresholdId::Sigmoid>(x); });
  }

  template <>
  inline MatrixXd evaluate_0_mat<ThresholdId::ReLu>(const MatrixXd &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_0<ThresholdId::ReLu>(x); });
  }

  template <>
  inline MatrixXd evaluate_1_mat<ThresholdId::ReLu>(const MatrixXd &mat) {
    return mat.unaryExpr(
        [](double x) { return evaluate_1<ThresholdId::ReLu>(x); });
  }

  template <>
  inline MatrixXd evaluate_0_mat<ThresholdId::SoftMax>(const MatrixXd &mat) {
    MatrixXd res(mat.rows(), mat.cols());
    for (int i = 0; i < mat.cols(); ++i) {
      VectorXd exp_vec =
          mat.col(i).unaryExpr([](double el) { return exp(el); });
      double exp_sum = exp_vec.transpose() * VectorXd::Ones(exp_vec.rows());
      res.col(i) = exp_vec / exp_sum;
    }
    return res;
  }

  template <>
  inline MatrixXd evaluate_1_mat<ThresholdId::SoftMax>(const MatrixXd &mat) {
    MatrixXd res(mat.rows(), mat.cols());
    for (Index i = 0; i < mat.cols(); ++i) {
      double max_coeff = mat.col(i).maxCoeff();
      VectorXd exp_vec = mat.col(i).unaryExpr(
          [max_coeff](double el) { return exp(el - max_coeff); });
      assert(std::isfinite(exp_vec.maxCoeff()));
      double exp_sum = exp_vec.transpose() * VectorXd::Ones(exp_vec.rows());
      VectorXd c = exp_sum * VectorXd::Ones(exp_vec.rows()) - exp_vec;
      res.col(i) = c.cwiseProduct(exp_vec / (exp_sum * exp_sum));
      assert(isfinite(res.col(i).maxCoeff()));
    }
    return res;
  }
};

class ThresholdFunc {

public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  using FunctionType = std::function<MatrixXd(const MatrixXd &)>;
  ThresholdFunc(FunctionType evaluate_0, FunctionType evaluate_1);

  template <ThresholdId Id> static ThresholdFunc create() {
    return ThresholdFunc(ThresholdDatabase::evaluate_0_mat<Id>,
                         ThresholdDatabase::evaluate_1_mat<Id>);
  }

  static ThresholdFunc create(ThresholdId threshold);

  MatrixXd apply(const MatrixXd &layer_val) const;

  MatrixXd derive(const MatrixXd &layer_val) const;

  bool check_empty();

private:
  FunctionType evaluate_0_;
  FunctionType evaluate_1_;
};
} // namespace network
