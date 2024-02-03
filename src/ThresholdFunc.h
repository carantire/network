#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>

namespace network {

enum class ThresholdId { Sigmoid, ReLu, SoftMax };

struct ThresholdDatabase {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  template <ThresholdId> static double evaluate_0(double);

  template <ThresholdId> static MatrixXd evaluate_0_mat(const MatrixXd &);

  template <ThresholdId> static double evaluate_1(double);

  template <ThresholdId> static MatrixXd evaluate_1_mat(const MatrixXd &);

  template <> inline double evaluate_0<ThresholdId::Sigmoid>(double x) {
    return 1. / (1. + std::exp(-x));
  }

  template <> inline double evaluate_1<ThresholdId::Sigmoid>(double x) {
    return 1. / (std::exp(-x) + std::exp(x) + 2.);
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
      VectorXd exp_vec =
          mat.col(i).unaryExpr([](double el) { return exp(el); });
      res.col(i) =
          exp_vec / (VectorXd::Ones(exp_vec.rows()).transpose() * exp_vec);
    }
    return res;
  }

  template <>
  inline MatrixXd evaluate_1_mat<ThresholdId::SoftMax>(const MatrixXd &mat) {
    MatrixXd res(mat.rows(), mat.cols());
    for (int i = 0; i < mat.cols(); ++i) {
      VectorXd exp_col =
          mat.col(i).unaryExpr([](double el) { return exp(el); });
      double exp_sum = VectorXd::Ones(mat.rows()).transpose() * exp_col;
      VectorXd const_vec = exp_sum * VectorXd::Ones(mat.rows()) - exp_col;
      res.col(i) = const_vec.cwiseProduct(exp_col) / (exp_sum * exp_sum);
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
