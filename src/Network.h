#pragma once

#include "Layer.h"
#include "ScoreFunc.h"
#include <Eigen/Eigen>
#include <iostream>

namespace network {

enum class LearningSpeedId {Const, Linear, Exponent};

struct LayerValue {
  using MatrixXd = Eigen::MatrixXd;
  MatrixXd in;
  MatrixXd out;
};

class Network {
  template <class T> using vector = std::vector<T>;

public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  template <class T> using vector = std::vector<T>;
  using Index = Eigen::Index;

  Network(std::initializer_list<int> dimensions,
          std::initializer_list<ThresholdId> threshold_id, int seed, double normalize);

  VectorXd Calculate(const VectorXd &start_vec) const;

  void Train(const MatrixXd &input, const MatrixXd &target,
             const ScoreFunc &score_func, int epoch_num, int batch_size);


  vector<Layer> layers_;

private:
  vector<LayerValue> Forward_Prop(const MatrixXd &start_vec) const;
  double Back_Prop(const vector<LayerValue> &layer_values,
                   const MatrixXd &target, const ScoreFunc &score_func,
                   double step);
  MatrixXd GetGradMatrix(const MatrixXd &input, const MatrixXd &target,
                         const ScoreFunc &score_func) const;
};
} // namespace network
