//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Score_Func.h"

Score_Func::Score_Func(ScoreType score_func, GradientType gradient_func) : score_func_(std::move(score_func)),
gradient_func_(std::move(gradient_func)) {}


Score_Func Score_Func::create(Score_Id score) {
    switch (score) {
        default:
            return create<Score_Id::MSE>();
    }
}

double Score_Func::score(const Eigen::VectorXd &x, const Eigen::VectorXd &reference) const {
    return score_func_(x, reference);
}

Eigen::VectorXd Score_Func::gradient(const Eigen::VectorXd &x, const Eigen::VectorXd &reference) const {
    return gradient_func_(x, reference);
}
