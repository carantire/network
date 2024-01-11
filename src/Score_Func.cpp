//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Score_Func.h"
using Eigen::VectorXd;

namespace network {

Score_Func::Score_Func(ScoreFuncType score_func, GradientFuncType gradient_func)
    : score_func_(std::move(score_func)), gradient_func_(std::move(gradient_func)) {}

Score_Func Score_Func::create(Score_Id score) {
    switch (score) {
    default:
        return create<Score_Id::MSE>();
    }
}

double Score_Func::score(const VectorXd &x, const VectorXd &reference) const {
    return score_func_(x, reference);
}

VectorXd Score_Func::gradient(const VectorXd &x, const VectorXd &reference) const {
    return gradient_func_(x, reference);
}
} // namespace network
