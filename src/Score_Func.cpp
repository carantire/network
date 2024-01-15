//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Score_Func.h"
using Eigen::VectorXd;

namespace network {

Score_Func::Score_Func(ScoreFuncType score_func, GradientFuncType gradient_func)
    : score_func_(std::move(score_func)),
      gradient_func_(std::move(gradient_func)) {
  assert(score_func_ && "Empty score function!");
  assert(gradient_func_ && "Empty gradient function!");
}

Score_Func Score_Func::create(Score_Id score) {
  switch (score) {
  case Score_Id::MSE: {
    return create<Score_Id::MSE>();
  }
  case Score_Id::MAE: {
    return create<Score_Id::MAE>();
  }
  case Score_Id::CrossEntropy: {
    return create<Score_Id::CrossEntropy>();
  }
  default:
    return create<Score_Id::MSE>();
  }
}

double Score_Func::score(const VectorXd &x, const VectorXd &reference) const {
  assert(score_func_ && "Empty score function!");
  return score_func_(x, reference);
}

VectorXd Score_Func::gradient(const VectorXd &x,
                              const VectorXd &reference) const {
  assert(gradient_func_ && "Empty gradient function!");
  return gradient_func_(x, reference);
}
VectorXd Score_Database::SoftMax(const VectorXd &vec) {
  VectorXd exp_vec = vec.array().exp();
  return exp_vec / exp_vec.sum();
}

} // namespace network
