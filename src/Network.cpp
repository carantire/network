//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Network.h"
namespace network {

Network::Network(std::vector<int> dimensions, std::vector<Threshold_Id> threshold_funcs)
    : dimensions_(std::move(dimensions)), threshold_funcs_(std::move(threshold_funcs)) {
    layers.reserve(dimensions_.size() - 1);
    for (size_t i = 0; i + 1 < dimensions_.size(); ++i) {
        layers.emplace_back(Layer(threshold_funcs_[i], dimensions_[i], dimensions_[i + 1]));
    }
}

Eigen::VectorXd Network::Forward_Prop(const Eigen::VectorXd &start_vec) {
    Eigen::VectorXd cur_vec = start_vec;
    for (size_t i = 0; i < layers.size(); ++i) {
        in_values[i] = cur_vec;
        cur_vec = layers[i].apply(cur_vec);
        out_values[i] = cur_vec;
        cur_vec = Threshold_Func::create(threshold_funcs_[i]).apply(cur_vec);
    }
    return cur_vec;
}

Eigen::VectorXd Network::Back_Prop(const Eigen::VectorXd &start_vec, const Eigen::VectorXd &reference,
                                   const Score_Func &score_func, double coef) {
    Eigen::VectorXd finish_vec = Forward_Prop(start_vec);
    Eigen::VectorXd u = score_func.gradient(finish_vec, reference);
    for (size_t i = layers.size() - 1; i != 0; --i) {
        layers[i].apply_gradA(layers[i].gradA(in_values[i], u, out_values[i]), coef);
        layers[i].apply_gradb(layers[i].gradb(u, out_values[i]), coef);
        u = layers[i].gradx(u, out_values[i]);
    }
    return u;
}
} // namespace network