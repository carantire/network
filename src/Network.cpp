//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Network.h"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;
namespace network {

Network::Network(std::initializer_list<int> dimensions, std::initializer_list<Threshold_Id> threshold_id): threshold_id_(threshold_id) {
    layers_.reserve(dimensions.size() - 1);
    auto dim_it = dimensions.begin();
    for (auto threshold_it = threshold_id.begin(); threshold_it != threshold_id.end(); ++dim_it, ++threshold_it) {
        layers_.emplace_back(*threshold_it, *std::next(dim_it), *dim_it);
    }
}

Values Network::Forward_Prop(const VectorXd &start_vec) {
    Values values;
    values.in.reserve(layers_.size());
    values.in.resize(layers_.size());
    values.out.reserve(layers_.size());
    values.out.resize(layers_.size());
    VectorXd cur_vec = start_vec;
    for (size_t i = 0; i < layers_.size(); ++i) {
        values.in[i] = cur_vec;
        cur_vec = layers_[i].apply(cur_vec);
        values.out[i] = cur_vec;
        cur_vec = Threshold_Func::create(threshold_id_[i]).apply(cur_vec);
    }
    return values;
}

VectorXd Network::Back_Prop(const VectorXd &start_vec, const VectorXd &reference, const Score_Func &score_func,
                            double step) {
    Values values = Forward_Prop(start_vec);
    const auto func = Threshold_Func::create(threshold_id_.back());
    VectorXd u = score_func.gradient(func.apply(values.out.back()), reference);
    for (int i = layers_.size() - 1; i >= 0; --i) {
        layers_[i].apply_gradA(layers_[i].gradA(values.in[i], u, values.out[i]), step);
        layers_[i].apply_gradb(layers_[i].gradb(u, values.out[i]), step);
        u = layers_[i].gradx(u, values.out[i]);
    }
    return u;
}
} // namespace network
