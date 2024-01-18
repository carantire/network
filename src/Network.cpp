//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Network.h"

namespace network {

using VectorXd = Network::VectorXd;
using PermutationMatrix = Network::PermutationMatrix;

Network::Network(std::initializer_list<int> dimensions,
                 std::initializer_list<Threshold_Id> threshold_id)
    : threshold_id_(threshold_id) {
  layers_.reserve(dimensions.size() - 1);
  auto dim_it = dimensions.begin();
  for (auto threshold_it = threshold_id.begin();
       threshold_it != threshold_id.end(); ++dim_it, ++threshold_it) {
    layers_.emplace_back(*threshold_it, *std::next(dim_it), *dim_it);
  }
}

Values Network::Forward_Prop(const VectorXd &start_vec) {
  Values values;
  values.in.resize(layers_.size());
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

VectorXd Network::Apply(const VectorXd &start_vec) {
  auto cur_vec = start_vec;
  for (size_t i = 0; i < layers_.size(); ++i) {
    cur_vec = layers_[i].apply(cur_vec);
    cur_vec = Threshold_Func::create(threshold_id_[i]).apply(cur_vec);
  }
  return cur_vec;
}

VectorXd Network::Back_Prop(const vector<Values> &values,
                            const MatrixXd &reference,
                            const Score_Func &score_func, double step) {
  const auto func = Threshold_Func::create(threshold_id_.back());
  MatrixXd u(reference.rows(), reference.cols());
  for (int i = 0; i < u.cols(); ++i) {
    u.col(i) =
        score_func.gradient(func.apply(values[i].out.back()), reference.col(i));
  }

  int dim = values.size();
  VectorXd grad;
  for (int i = layers_.size() - 1; i >= 0; --i) {
    VectorXd delta_in = VectorXd::Zero(values[0].in[i].rows());
    VectorXd delta_out = VectorXd::Zero(values[0].out[i].rows());
    MatrixXd new_u(layers_[i].Get_Dim(), u.cols());
    grad = VectorXd::Zero(u.rows());
    for (int j = 0; j < dim; ++j) {
      delta_in += values[j].in[i] / dim;
      delta_out += values[j].out[i] / dim;
      grad += u.col(j) / dim;
      new_u.col(j) = layers_[i].gradx(u.col(j), delta_out);
    }
    layers_[i].apply_gradA(layers_[i].gradA(delta_in, grad, delta_out), step);
    layers_[i].apply_gradb(layers_[i].gradb(grad, delta_out), step);
    u = new_u;
  }

  return grad;
}
VectorXd Network::Back_Prop_SGD(const MatrixXd &start_batch,
                                const MatrixXd &reference,
                                const Score_Func &score_func, int iter_num) {
  auto rand_ind = index_generator_() % start_batch.cols();
  auto values = {Forward_Prop(start_batch.col(rand_ind))};
  return Back_Prop(values, reference.col(rand_ind), score_func, 1.0 / iter_num);
}
void Network::TrainSGD(const MatrixXd &start_batch, const MatrixXd &reference,
                       const Score_Func &score_func, double needed_accuracy,
                       int max_epochs) {
  int epochs = 0;
  double cur_acc = 1e9;
  vector<Values> values(start_batch.cols());
  while (epochs != max_epochs && cur_acc >= needed_accuracy) {
    for (int i = 0; i < start_batch.cols(); ++i) {
      values[i] = Forward_Prop(start_batch.col(i));
    }
    cur_acc = Back_Prop(values, reference, score_func, epochs)
                  .unaryExpr([](double x) { return abs(x); })
                  .maxCoeff();
    ++epochs;
  }
}

void Network::TrainBGD(const MatrixXd &start_batch, const MatrixXd &reference,
                       const Score_Func &score_func, int cols_in_minibatch,
                       double needed_accuracy, int max_epochs) {
  int epochs = 0;
  double cur_acc = 1e9;
  vector<Values> values(start_batch.cols());
  auto cur_batch = start_batch;
  auto cur_ref = reference;
  while (epochs != max_epochs && cur_acc >= needed_accuracy) {
    for (int i = 0; i < start_batch.cols(); ++i) {
      values[i] = Forward_Prop(cur_batch.col(i));
    }
    cur_acc = Back_Prop(values, cur_ref, score_func, epochs)
                  .unaryExpr([](double x) { return abs(x); })
                  .maxCoeff();
    cur_batch *= GetRandMat(cur_batch.cols());
    cur_ref *= GetRandMat(cur_ref.cols());
    ++epochs;
  }
}

PermutationMatrix Network::GetRandMat(int cols) {
  PermutationMatrix perm(cols);
  perm.setIdentity();
  std::shuffle(perm.indices().data(),
               perm.indices().data() + perm.indices().size(), index_generator_);
  return perm;
}

} // namespace network
