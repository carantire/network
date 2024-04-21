#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

enum class ScoreId { MSE, MAE, CrossEntropy };

class ScoreFunc {

  template <ScoreId Id> static ScoreFunc create();

public:
  using VectorXd = Eigen::VectorXd;
  using MatrixXd = Eigen::MatrixXd;
  using ScoreFuncType =
      std::function<double(const VectorXd &, const VectorXd &)>;
  using GradientFuncType =
      std::function<MatrixXd(const MatrixXd &, const MatrixXd &)>;

  ScoreFunc(ScoreFuncType score_func, GradientFuncType gradient_func);

  static ScoreFunc create(ScoreId score);

  double score(const VectorXd &x, const VectorXd &reference) const;

  MatrixXd gradient(const MatrixXd &x, const MatrixXd &reference) const;

  bool check_empty();

private:
  ScoreFuncType score_func_;
  GradientFuncType gradient_func_;
};
} // namespace network
