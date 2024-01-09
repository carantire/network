#pragma once

#include "Layer.h"
#include "Score_Func.h"
#include <Eigen/Eigen>
#include <EigenRand/EigenRand/EigenRand>
#include <cmath>
#include <iostream>
#include <utility>

namespace network {
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

class Network {
    Network(vector<int> dimensions, vector<Threshold_Id> threshold_funcs);

    VectorXd Forward_Prop(const VectorXd &start_vec);

    Eigen::VectorXd Back_Prop(const VectorXd &start_vec, const VectorXd &reference, const Score_Func &score_func,
                              double coef);

    private:
    vector<Layer> layers;
    vector<int> dimensions_;
    vector<Threshold_Id> threshold_funcs_;
    vector<VectorXd> in_values;
    vector<VectorXd> out_values;
};
} // namespace network