//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "ScoreFunc.h"

using Eigen::VectorXd;

namespace network {
struct ScoreDatabase {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  static VectorXd SoftMax(const VectorXd &vec);

  template <ScoreId> static double score(const VectorXd &, const VectorXd &);

  template <ScoreId>
  static VectorXd gradient(const VectorXd &, const VectorXd &);

  template <>
  inline double score<ScoreId::MSE>(const VectorXd &input,
                                    const VectorXd &target) {
    return (input - target).dot(input - target);
  }

  template <>
  inline VectorXd gradient<ScoreId::MSE>(const VectorXd &x,
                                         const VectorXd &reference) {
    return 2.0 * (x - reference);
  }

  template <>
  inline double score<ScoreId::MAE>(const VectorXd &input,
                                    const VectorXd &target) {
    return VectorXd::Ones(input.rows()).transpose() *
           (input - target).unaryExpr([](double el) { return abs(el); });
  }

  template <>
  inline VectorXd gradient<ScoreId::MAE>(const VectorXd &input,
                                         const VectorXd &target) {
    return (input - target).unaryExpr([](double el) {
      return el > 0 ? 1.0 : -1.0;
    });
  }
  template <>
  inline double score<ScoreId::CrossEntropy>(const VectorXd &input,
                                             const VectorXd &target) {
    return -SoftMax(target).transpose() *
           SoftMax(input).unaryExpr([](double el) { return log(el); });
  }
  template <>
  inline VectorXd gradient<ScoreId::CrossEntropy>(const VectorXd &input,
                                                  const VectorXd &target) {
    double exp_sum = VectorXd::Ones(input.rows()).transpose() *
                     input.unaryExpr([](double el) { return exp(el); });
    auto const_vec = input.unaryExpr([](double el) { return exp(el); }) -
                     exp_sum * VectorXd::Ones(input.rows());
    return SoftMax(target).asDiagonal() * const_vec / exp_sum;
  }
};

ScoreFunc::ScoreFunc(ScoreFuncType score_func, GradientFuncType gradient_func)
    : score_func_(std::move(score_func)),
      gradient_func_(std::move(gradient_func)) {
  assert(score_func_ && "Empty score function!");
  assert(gradient_func_ && "Empty gradient function!");
}

template <ScoreId Id> ScoreFunc ScoreFunc::create() {
  return ScoreFunc(ScoreDatabase::score<Id>, ScoreDatabase::gradient<Id>);
}

ScoreFunc ScoreFunc::create(ScoreId score) {
  switch (score) {
  case ScoreId::MSE: {
    return create<ScoreId::MSE>();
  }
  case ScoreId::MAE: {
    return create<ScoreId::MAE>();
  }
  case ScoreId::CrossEntropy: {
    return create<ScoreId::CrossEntropy>();
  }
  default:
    return create<ScoreId::MSE>();
  }
}

double ScoreFunc::score(const VectorXd &input, const VectorXd &target) const {
  return score_func_(input, target);
}

VectorXd ScoreFunc::gradient(const VectorXd &input,
                             const VectorXd &target) const {
  return gradient_func_(input, target);
}

VectorXd ScoreDatabase::SoftMax(const VectorXd &vec) {
  VectorXd exp_vec = vec.unaryExpr([](double el) { return exp(el); });
  return exp_vec / (VectorXd::Ones(exp_vec.rows()).transpose() * exp_vec);
}

bool ScoreFunc::check_empty() { return score_func_ && gradient_func_; }

} // namespace network
