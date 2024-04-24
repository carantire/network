#include "Network.h"
#include "ThresholdFunc.h"

namespace network {

Network::Network(std::initializer_list<int> dimensions,
                 std::initializer_list<ThresholdId> threshold_id, int seed,
                 double normalize) {
  assert(dimensions.size() == threshold_id.size() + 1);
  layers_.reserve(dimensions.size() - 1);
  auto dim_it = dimensions.begin();
  for (auto threshold_it = threshold_id.begin();
       threshold_it != threshold_id.end(); ++dim_it, ++threshold_it) {
    layers_.emplace_back(*threshold_it, *dim_it, *std::next(dim_it), seed,
                         normalize);
  }
}

std::vector<LayerValue> Network::Forward_Prop(const Matrix &start_mat) const {
  std::vector<LayerValue> layer_values(layers_.size());
  Matrix cur_mat = start_mat;
  for (Index i = 0; i < layers_.size(); ++i) {
    layer_values[i].in = cur_mat;
    cur_mat = layers_[i].apply_linear(cur_mat);
    layer_values[i].out = cur_mat;
    cur_mat = layers_[i].apply_threshold(cur_mat);
  }
  return layer_values;
}

Vector Network::Calculate(const Vector &start_vec) const {
  assert(start_vec.rows() == layers_.front().Get_Input_Dim());
  auto cur_mat = start_vec;
  for (size_t i = 0; i < layers_.size(); ++i) {
    cur_mat = layers_[i].apply_linear(cur_mat);
    cur_mat = layers_[i].apply_threshold(cur_mat);
  }
  return cur_mat;
}

void Network::Train(const Matrix &input, const Matrix &target,
                    const ScoreFunc &score_func,
                    const LearningRate &learning_rate, int epoch_num,
                    int batch_size) {
  assert(input.cols() == target.cols() &&
         "Target size must coincide with batch size");
  assert(input.cols() % batch_size == 0 && "Number of batches must be integer");
  for (Index epoch = 1; epoch <= epoch_num; ++epoch) {
    std::cout << "Epoch num: " << epoch << '\n';
    for (Index batch_num = 0; batch_num < input.cols() / batch_size;
         ++batch_num) {
      Index start_ind = batch_num * batch_size;
      const Matrix &input_batch =
          input.block(0, start_ind, input.rows(), batch_size);
      const Matrix &output_batch =
          target.block(0, start_ind, target.rows(), batch_size);
      Back_Prop(Forward_Prop(input_batch), output_batch, score_func,
                learning_rate(epoch));
    }
  }
}

double Network::Back_Prop(const std::vector<LayerValue> &layer_values,
                          const Matrix &target, const ScoreFunc &score_func,
                          double step) {
  auto grad = GetGradMatrix(layer_values.back().out, target, score_func);
  for (int i = layers_.size() - 1; i >= 0; --i) {
    layers_[i].apply_gradA(layer_values[i].in, grad, layer_values[i].out, step);
    layers_[i].apply_gradb(grad, layer_values[i].out, step);
    grad = layers_[i].gradx(grad, layer_values[i].out);
  }
  return Vector::Ones(grad.rows()).transpose() *
         (grad * Vector::Ones(grad.cols()) / grad.cols())
             .unaryExpr([](double el) { return abs(el); });
}

Matrix Network::GetGradMatrix(const Matrix &input, const Matrix &target,
                              const ScoreFunc &score_func) const {

  auto final_mat = layers_.back().apply_threshold(input);
  assert(final_mat.size() == target.size());
  Matrix res = score_func.gradient(final_mat, target);
  return res;
}

} // namespace network
