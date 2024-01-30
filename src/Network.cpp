//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Network.h"
#include "ThresholdFunc.h"

namespace network {

using VectorXd = Network::VectorXd;
using MatrixXd = Network::MatrixXd;
using vector = Network::vector<LayerValue>;

Network::Network(std::initializer_list<int> dimensions,
                 std::initializer_list<ThresholdId> threshold_id) {
  if (dimensions.size() != threshold_id.size() + 1) {
    throw std::invalid_argument("Invalid network constructor input");
  }
  layers_.reserve(dimensions.size() - 1);
  auto dim_it = dimensions.begin();
  for (auto threshold_it = threshold_id.begin();
       threshold_it != threshold_id.end(); ++dim_it, ++threshold_it) {
    layers_.emplace_back(*threshold_it, *dim_it, *std::next(dim_it));
  }
}

std::vector<LayerValue> Network::Forward_Prop(const MatrixXd &start_mat) const {
  vector<LayerValue> layer_values(layers_.size());
  MatrixXd cur_mat = start_mat;
  for (size_t i = 0; i < layers_.size(); ++i) {
    layer_values[i].in = cur_mat;
    cur_mat = layers_[i].apply_linear(cur_mat);
    layer_values[i].out = cur_mat;
    cur_mat = layers_[i].apply_threshold(cur_mat);
  }
  return layer_values;
}

VectorXd Network::Calculate(const VectorXd &start_vec) const {
  if (start_vec.rows() != layers_.front().Get_Input_Dim()) {
    throw std::invalid_argument(
        "Vector size must coincide with first layer size");
  }
  auto cur_mat = start_vec;
  for (size_t i = 0; i < layers_.size(); ++i) {
    cur_mat = layers_[i].apply_linear(cur_mat);
    cur_mat = layers_[i].apply_threshold(cur_mat);
  }
  return cur_mat;
}

void Network::Train(const MatrixXd &start_batch, const MatrixXd &target,
                    const ScoreFunc &score_func, size_t max_epochs,
                    double accuracy) {
  if (start_batch.cols() != target.cols()) {
    throw std::length_error("Target size must coincide with batch size");
  }
  size_t epochs = 0;
  size_t bias = std::numeric_limits<size_t>::max();
  while (epochs != max_epochs && bias > accuracy) {
    bias =
        Back_Prop(Forward_Prop(start_batch), target, score_func, 1. / epochs);
    ++epochs;
  }
}

double Network::Back_Prop(const std::vector<LayerValue> &layer_values,
                          const MatrixXd &target, const ScoreFunc &score_func,
                          double step) {
  auto grad = GetGradMatrix(layer_values.back().out, target, score_func);
  for (int i = layers_.size() - 1; i >= 0; --i) {
    layers_[i].apply_gradA(layer_values[i].in, grad, layer_values[i].out, step);
    layers_[i].apply_gradb(grad, layer_values[i].out, step);
    grad = layers_[i].gradx(grad, layer_values[i].out);
  }
  return VectorXd::Ones(grad.rows()).transpose() *
         (grad * VectorXd::Ones(grad.cols()) / grad.cols())
             .unaryExpr([](double el) { return abs(el); });
}

MatrixXd Network::GetGradMatrix(const MatrixXd &input, const MatrixXd &target,
                                const ScoreFunc &score_func) const {
  MatrixXd res(target.rows(), target.cols());
  auto final_mat = layers_.back().apply_threshold(input);
  for (Index i = 0; i < res.cols(); ++i) {
    res.col(i) = score_func.gradient(final_mat.col(i), target.col(i));
  }
  return res;
}

} // namespace network
