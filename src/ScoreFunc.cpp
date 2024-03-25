//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "ScoreFunc.h"

using Eigen::MatrixXd;

namespace network {
struct ScoreDatabase {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  template <ScoreId> static double score(const VectorXd &, const VectorXd &);

  template <ScoreId>
  static MatrixXd gradient(const MatrixXd &, const MatrixXd &);

  template <>
  inline double score<ScoreId::MSE>(const VectorXd &input,
                                    const VectorXd &target) {
    return (input - target).dot(input - target);
  }

  template <>
  inline MatrixXd gradient<ScoreId::MSE>(const MatrixXd &input,
                                         const MatrixXd &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return 2.0 * (input - target);
  }

  template <>
  inline double score<ScoreId::MAE>(const VectorXd &input,
                                    const VectorXd &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return VectorXd::Ones(input.rows()).transpose() *
           (input - target).unaryExpr([](double el) { return abs(el); });
  }

  template <>
  inline MatrixXd gradient<ScoreId::MAE>(const MatrixXd &input,
                                         const MatrixXd &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return (input - target).unaryExpr([](double el) {
      return el > 0 ? 1.0 : -1.0;
    });
  }
  template <>
  inline double score<ScoreId::CrossEntropy>(const VectorXd &input,
                                             const VectorXd &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return -target.transpose() *
           input.unaryExpr([](double el) { return log(el); });
  }
  template <>
  inline MatrixXd gradient<ScoreId::CrossEntropy>(const MatrixXd &input,
                                                  const MatrixXd &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return -target.cwiseProduct(
        input.unaryExpr([](double el) { return 1.0 / el; }));
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

Eigen::MatrixXd ScoreFunc::gradient(const MatrixXd &input,
                                    const MatrixXd &target) const {
  return gradient_func_(input, target);
}

bool ScoreFunc::check_empty() { return score_func_ && gradient_func_; }

} // namespace network