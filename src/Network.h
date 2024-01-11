#pragma once

#include "Layer.h"
#include "Score_Func.h"
#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>
#include <iostream>
#include <utility>

namespace network {

struct Values {
    using VectorXd = Eigen::VectorXd;
    template <class T>
    using vector = std::vector<T>;
    vector<VectorXd> in;
    vector<VectorXd> out;
};

class Network {
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    template <class T>
    using vector = std::vector<T>;

    public:
    Network(std::initializer_list<int> dimensions, std::initializer_list<Threshold_Id> threshold_id);

    Values Forward_Prop(const VectorXd &start_vec);

    VectorXd Back_Prop(const VectorXd &start_vec, const VectorXd &reference, const Score_Func &score_func, double coef);

    private:
    vector<Layer> layers_;
    vector<int> dimensions_;
    vector<Threshold_Id> threshold_id_;
};
} // namespace network
