#pragma once

#include <iostream>
#include <cmath>
#include <utility>
#include "Eigen/Dense"
#include "EigenRand/EigenRand"
#include "Layer.h"
#include "Score_Func.h"

class Network {
    Network(std::vector<int> dimensions, std::vector<Threshold_Id> threshold_funcs);

    Eigen::VectorXd Forward_Prop(const Eigen::VectorXd &start_vec);

    Eigen::VectorXd
    Back_Prop(const Eigen::VectorXd &start_vec, const Eigen::VectorXd &reference, const Score_Func &score_func,
              double coef);


private:
    std::vector<Layer> layers;
    std::vector<int> dimensions_;
    std::vector<Threshold_Id> threshold_funcs_;
    std::vector<Eigen::VectorXd> in_values;
    std::vector<Eigen::VectorXd> out_values;
};
