#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>
#include <iostream>

namespace network {

enum class ThresholdId { Sigmoid, ReLu, SoftMax };

struct ThresholdDatabase {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  using Index = Eigen::Index;

  template <ThresholdId> static double evaluate_0(double);

  template <ThresholdId> static MatrixXd evaluate_0_mat(const MatrixXd &);

  template <ThresholdId> static double evaluate_1(double);

  template <ThresholdId> static MatrixXd evaluate_1_mat(const MatrixXd &);

  template <> inline double evaluate_0<ThresholdId::Sigmoid>(double x) {
    std::cout << x << std::endl;
    assert(std::isfinite(x));
    double result = 1 / (1 + std::exp(x));
    assert(std::isfinite(result));
    return result;
  }

  template <> inline double evaluate_1<ThresholdId::Sigmoid>(double x) {
    std::cout << x << std::endl;
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
      for (int j = 0; j < mat.rows(); ++j) {
        res(j, i) = 1. / (VectorXd::Ones(mat.rows()).transpose() *
                          mat.col(i).unaryExpr([&](double el) {
                            return exp(std::min(28., el - mat(j, i)));
                          }));
      }
    }
    return res;
  }

  template <>
  inline MatrixXd evaluate_1_mat<ThresholdId::SoftMax>(const MatrixXd &mat) {
    MatrixXd res(mat.rows(), mat.cols());
    for (Index i = 0; i < mat.cols(); ++i) {
      for (int j = 0; j < mat.rows(); ++j) {
        double exp_diff_sum = 1. / (VectorXd::Ones(mat.rows()).transpose() *
                                    mat.col(i).unaryExpr([&](double el) {
                                      return exp(std::min(28., el - mat(j, i)));
                                    }));
        res(j, i) = exp_diff_sum * (1 - exp_diff_sum);
      }
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
