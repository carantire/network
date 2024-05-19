#include "Network.h"
#include "ThresholdFunc.h"

namespace network {

Network::Network(std::initializer_list<int> dimensions,
                 std::initializer_list<ThresholdId> threshold_id, int seed,
                 double normalize) {
  assert(dimensions.size() == threshold_id.size() + 1);
  layers_.reserve(dimensions.size() - 1);
  auto dim_it = dimensions.begin();
  RandGen rng = seed;
  for (auto threshold_it = threshold_id.begin();
       threshold_it != threshold_id.end(); ++dim_it, ++threshold_it) {
    layers_.emplace_back(*threshold_it, *dim_it, *std::next(dim_it), rng,
                         normalize);
  }
}

Network::Network(std::vector<Layer> &&layers) : layers_(layers){};

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

void Network::ShuffleData(Matrix &input, Matrix &target, RandGen &rng) {
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic> perm(input.cols());
  perm.setIdentity();
  std::shuffle(perm.indices().data(),
               perm.indices().data() + perm.indices().size(),
               Eigen::Rand::PacketRandomEngineAdaptor<uint64_t, RandGen>(rng));
  input *= perm;
  target *= perm;
}

void Network::Train_GD(Matrix input, Matrix target, const ScoreFunc &score_func,
                       const LearningRate &learning_rate, int epoch_num,
                       int batch_size, int seed) {
  assert(input.cols() == target.cols() &&
         "Target size must coincide with batch size");
  assert(input.cols() % batch_size == 0 && "Number of batches must be integer");
  RandGen rng = seed;
  for (Index epoch = 1; epoch <= epoch_num; ++epoch) {
    std::cout << "Epoch num: " << epoch << '\n';
    ShuffleData(input, target, rng);
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

void Network::Train_SGD(Matrix input, Matrix target,
                        const ScoreFunc &score_func,
                        const LearningRate &learning_rate, int epoch_num,
                        int sample_size, int seed) {
  assert(input.cols() == target.cols() &&
         "Target size must coincide with batch size");
  assert(sample_size <= input.cols() &&
         "Sample size must be less or equal to dataset size");
  RandGen rng = seed;
  for (Index epoch = 1; epoch <= epoch_num; ++epoch) {
    std::cout << "Epoch num: " << epoch << '\n';
    ShuffleData(input, target, rng);
    for (Index el_num = 0; el_num < sample_size; ++el_num) {
      Index start_ind = el_num;
      const Matrix &input_el = input.block(0, start_ind, input.rows(), 1);
      const Matrix &output_el = target.block(0, start_ind, target.rows(), 1);
      Back_Prop(Forward_Prop(input_el), output_el, score_func,
                learning_rate(epoch));
    }
  }
}

void Network::StoreModel(const std::filesystem::path &path) {
  auto out_file = std::ofstream(
      path, std::ios_base::binary | std::ios_base::out | std::ios_base::trunc);
  uint32_t layers_num = layers_.size();
  out_file.write(reinterpret_cast<char*>(&layers_num), sizeof(layers_num));
  for (const auto &layer : layers_) {
    layer.WriteParams(out_file);
  }
}

Network Network::LoadModel(const std::filesystem::path &path) {
  auto in_file = std::ifstream(path, std::ios_base::binary | std::ios_base::in);
  uint32_t layers_num;
  in_file.read(reinterpret_cast<char*>(&layers_num), sizeof(layers_num));
  std::vector<Layer> layers;
  layers.reserve(layers_num);
  for (uint32_t i = 0; i < layers_num; ++i) {
    layers.push_back(Layer::ReadParams(in_file));
  }
  return Network(std::move(layers));
}

void Network::Back_Prop(const std::vector<LayerValue> &layer_values,
                          const Matrix &target, const ScoreFunc &score_func,
                          double step) {
  auto grad = GetGradMatrix(layer_values.back().out, target, score_func);
  for (Index i = layers_.size() - 1; i >= 0; --i) {
    layers_[i].apply_gradA(layer_values[i].in, grad, layer_values[i].out, step);
    layers_[i].apply_gradb(grad, layer_values[i].out, step);
    grad = layers_[i].gradx(grad, layer_values[i].out);
  }

}

Matrix Network::GetGradMatrix(const Matrix &input, const Matrix &target,
                              const ScoreFunc &score_func) const {

  auto final_mat = layers_.back().apply_threshold(input);
  if (final_mat.size() != target.size()) {
    std::cout << final_mat.rows() << " " << target.rows() << '\n';
  }
  assert(final_mat.size() == target.size());
  Matrix res = score_func.gradient(final_mat, target);
  return res;
}

} // namespace network
