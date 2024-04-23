#pragma once

#include "utils.h"
#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

enum class ScoreId { MSE, MAE, CrossEntropy };

class ScoreFunc {

  template <ScoreId Id> static ScoreFunc create();

public:
  using ScoreFuncType = std::function<double(const Vector &, const Vector &)>;
  using GradientFuncType =
      std::function<Matrix(const Matrix &, const Matrix &)>;

  ScoreFunc(ScoreFuncType score_func, GradientFuncType gradient_func);

  static ScoreFunc create(ScoreId score);

  double score(const Vector &x, const Vector &reference) const;

  Matrix gradient(const Matrix &x, const Matrix &reference) const;

  bool check_empty();

private:
  ScoreFuncType score_func_;
  GradientFuncType gradient_func_;
};
} // namespace network
