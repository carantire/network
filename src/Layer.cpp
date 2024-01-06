//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Layer.h"


Layer::Layer(Threshold_Id id, int rows, int columns) : threshold_func_(Threshold_Func::create(id)),
                                                       A_(Eigen::Rand::normal<Eigen::MatrixXd>(rows, columns, urng)),
                                                       b_(Eigen::Rand::normal<Eigen::VectorXd>(rows, 1, urng)) {
}

Eigen::VectorXd Layer::apply(const Eigen::VectorXd &x) const { // vector of values
    return threshold_func_.apply(A_ * x + b_);
}

Eigen::MatrixXd
Layer::derive(const Eigen::VectorXd &vec) const { // vec is a matrix of y_i = (Ax + b)_i - result of apply
    return threshold_func_.derive(vec).asDiagonal();
}

Eigen::MatrixXd
Layer::gradA(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
             const Eigen::VectorXd &vec) const { // u is a gradient vector
    return derive(vec) * u.transpose() * x.transpose();
}

Eigen::MatrixXd Layer::gradb(const Eigen::VectorXd &u, const Eigen::VectorXd &vec) const {
    return derive(vec) * u.transpose();
}

Eigen::VectorXd Layer::gradx(const Eigen::VectorXd &u, const Eigen::VectorXd &vec) const {
    return (A_.transpose() * derive(vec) * u.transpose()).transpose();
}

void Layer::apply_gradA(const Eigen::MatrixXd &grad, double coef) {
    A_ -= coef * grad;
}

void Layer::apply_gradb(const Eigen::VectorXd &grad, double coef) {
    b_ -= coef * grad;
}
