//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "ScoreFunc.h"

namespace network {

struct ScoreDatabase {
  template <ScoreId> static double score(const Vector &, const Vector &);

  template <ScoreId> static Matrix gradient(const Matrix &, const Matrix &);

  template <>
  inline double score<ScoreId::MSE>(const Vector &input, const Vector &target) {
    return (input - target).squaredNorm();
  }

  template <>
  inline Matrix gradient<ScoreId::MSE>(const Matrix &input,
                                       const Matrix &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return 2.0 * (input - target);
  }

  template <>
  inline double score<ScoreId::MAE>(const Vector &input, const Vector &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return
           (input - target).cwiseAbs().sum();
  }

  template <>
  inline Matrix gradient<ScoreId::MAE>(const Matrix &input,
                                       const Matrix &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return (input - target).cwiseSign();
  }
  template <>
  inline double score<ScoreId::CrossEntropy>(const Vector &input,
                                             const Vector &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return -target.transpose() *
           input.array().log().matrix();
  }
  template <>
  inline Matrix gradient<ScoreId::CrossEntropy>(const Matrix &input,
                                                const Matrix &target) {
    assert(input.rows() == target.rows() &&
           "input and target must have same size");
    return -target.cwiseProduct(
        input.cwiseInverse());
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

double ScoreFunc::score(const Vector &input, const Vector &target) const {
  return score_func_(input, target);
}

Matrix ScoreFunc::gradient(const Matrix &input, const Matrix &target) const {
  return gradient_func_(input, target);
}

bool ScoreFunc::check_empty() { return score_func_ && gradient_func_; }

} // namespace network
