#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

enum class ScoreId { MSE, MAE, CrossEntropy };

class ScoreFunc {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  using ScoreFuncType =
      std::function<double(const VectorXd &, const VectorXd &)>;
  using GradientFuncType =
      std::function<VectorXd(const VectorXd &, const VectorXd &)>;
  template <ScoreId Id> static ScoreFunc create();

public:
  ScoreFunc(ScoreFuncType score_func, GradientFuncType gradient_func);

  static ScoreFunc create(ScoreId score);

  double score(const VectorXd &x, const VectorXd &reference) const;

  VectorXd gradient(const VectorXd &x, const VectorXd &reference) const;

  bool check_empty();

private:
  ScoreFuncType score_func_;
  GradientFuncType gradient_func_;
};
} // namespace network
