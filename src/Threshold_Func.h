#pragma once

#include <iostream>
#include <cmath>
#include <utility>
#include "Eigen/Dense"
#include "EigenRand/EigenRand"

enum class Threshold_Id {
    Sigmoid,
    ReLu
};

struct Threshold_Database {
    template<Threshold_Id>
    static double evaluate_0(double);

    template<Threshold_Id>
    static double evaluate_1(double);

    template<>
    inline double evaluate_0<Threshold_Id::Sigmoid>(double x) {
        return 1. / (1. + std::exp(-x));
    }

    template<>
    inline double evaluate_1<Threshold_Id::Sigmoid>(double x) {
        return std::exp(-x) * evaluate_0<Threshold_Id::Sigmoid>(x) * evaluate_0<Threshold_Id::Sigmoid>(x);
    }


    template<>
    inline double evaluate_0<Threshold_Id::ReLu>(double x) {
        return x > 0 ? x : 0;
    }

    template<>
    inline double evaluate_1<Threshold_Id::ReLu>(double x) {
        return x > 0 ? 1 : 0;
    }
};

class Threshold_Func {
    using FunctionType = std::function<double(double)>;
public:
    Threshold_Func(FunctionType evaluate_0, FunctionType evaluate_1);

    template<Threshold_Id Id>
    static Threshold_Func create() {
        return Threshold_Func(Threshold_Database::evaluate_0<Id>, Threshold_Database::evaluate_1<Id>);
    }

    static Threshold_Func create(Threshold_Id threshold);

    double evaluate_0(double x) const;

    double evaluate_1(double x) const;

    Eigen::VectorXd apply(const Eigen::VectorXd &vec) const;

    Eigen::VectorXd derive(const Eigen::VectorXd &vec) const;

private:
    FunctionType evaluate_0_;
    FunctionType evaluate_1_;
};
