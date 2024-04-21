#pragma once

#include "Layer.h"
#include "LearningRate.h"
#include "ScoreFunc.h"
#include <Eigen/Eigen>
#include <cmath>
#include <iostream>

namespace network {

struct LayerValue {
  using MatrixXd = Eigen::MatrixXd;
  MatrixXd in;
  MatrixXd out;
};

class Network {
public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  template <class T> using vector = std::vector<T>;
  using Index = Eigen::Index;

  Network(std::initializer_list<int> dimensions,
          std::initializer_list<ThresholdId> threshold_id, int seed,
          double normalize);

  VectorXd Calculate(const VectorXd &start_vec) const;

  void Train(const MatrixXd &input, const MatrixXd &target,
             const ScoreFunc &score_func, const LearningRate &learning_rate,
             int epoch_num, int batch_size);

private:
  vector<Layer> layers_;

  vector<LayerValue> Forward_Prop(const MatrixXd &start_vec) const;
  double Back_Prop(const vector<LayerValue> &layer_values,
                   const MatrixXd &target, const ScoreFunc &score_func,
                   double step);
  MatrixXd GetGradMatrix(const MatrixXd &input, const MatrixXd &target,
                         const ScoreFunc &score_func) const;
};
} // namespace network
