//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Layer.h"

namespace network {
using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;    

Layer::Layer(Threshold_Id id, int rows, int columns)
    : threshold_func_(Threshold_Func::create(id)), A_(Eigen::Rand::normal<MatrixXd>(rows, columns, urng)),
      b_(Eigen::Rand::normal<VectorXd>(rows, 1, urng)) {}

VectorXd Layer::apply(const VectorXd &x) const { // vector of values
    return threshold_func_.apply(A_ * x + b_);
}

MatrixXd Layer::derive(const VectorXd &vec) const { // vec is a matrix of y_i = (Ax + b)_i - result of apply
    return threshold_func_.derive(vec).asDiagonal();
}

MatrixXd Layer::gradA(const VectorXd &x, const VectorXd &u,
                      const VectorXd &vec) const { // u is a gradient vector
    return derive(vec) * u.transpose() * x.transpose();
}

MatrixXd Layer::gradb(const VectorXd &u, const VectorXd &vec) const { return derive(vec) * u.transpose(); }

VectorXd Layer::gradx(const VectorXd &u, const VectorXd &vec) const {
    return (A_.transpose() * derive(vec) * u.transpose()).transpose();
}

void Layer::apply_gradA(const MatrixXd &grad, double step) { A_ -= step * grad; }

void Layer::apply_gradb(const VectorXd &grad, double step) { b_ -= step * grad; }
} // namespace network
