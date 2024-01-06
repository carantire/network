#pragma once

#include "Threshold_Func.h"
#include <utility>
#include "Eigen/Dense"
#include "EigenRand/EigenRand"

class Layer {

public:

    Layer(Threshold_Id id, int rows, int columns);

    Eigen::VectorXd apply(const Eigen::VectorXd &x) const; // vector of values
    Eigen::MatrixXd derive(const Eigen::VectorXd &vec) const;// vec is a matrix of y_i = (Ax + b)_i - result of apply

    Eigen::MatrixXd gradA(const Eigen::VectorXd &x, const Eigen::VectorXd &u,
                          const Eigen::VectorXd &vec) const; // u is a gradient vector

    Eigen::MatrixXd gradb(const Eigen::VectorXd &u, const Eigen::VectorXd &vec) const;

    Eigen::VectorXd gradx(const Eigen::VectorXd &u, const Eigen::VectorXd &vec) const;

    void apply_gradA(const Eigen::MatrixXd &grad, double coef);

    void apply_gradb(const Eigen::VectorXd &grad, double coef);

private:
    static inline Eigen::Rand::Vmt19937_64 urng = 1;
    Threshold_Func threshold_func_;
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
};