#pragma once

#include "Threshold_Func.h"
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {


class Layer {
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    public:
    Layer(Threshold_Id id, int rows, int columns);

    VectorXd apply(const VectorXd &x) const;    // vector of values
    MatrixXd derive(const VectorXd &vec) const; // vec is a matrix of y_i = (Ax + b)_i - result of apply

    MatrixXd gradA(const VectorXd &x, const VectorXd &u,
                          const VectorXd &vec) const; // u is a gradient vector

    MatrixXd gradb(const VectorXd &u, const VectorXd &vec) const;

    VectorXd gradx(const VectorXd &u, const VectorXd &vec) const;

    void apply_gradA(const MatrixXd &grad, double step);

    void apply_gradb(const VectorXd &grad, double step);

    private:
    static inline Eigen::Rand::Vmt19937_64 urng = 1;
    Threshold_Func threshold_func_;
    MatrixXd A_;
    VectorXd b_;
};
} // namespace network
