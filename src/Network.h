#pragma once

#include "Layer.h"
#include "ScoreFunc.h"
#include <Eigen/Eigen>

namespace network {

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
          std::initializer_list<ThresholdId> threshold_id);

  VectorXd Calculate(const VectorXd &start_vec) const;

  void Train(const MatrixXd &start_batch, const MatrixXd &target,
             const ScoreFunc &score_func, size_t max_epochs, double accuracy);

private:
  vector<LayerValue> Forward_Prop(const MatrixXd &start_vec) const;
  double Back_Prop(const vector<LayerValue> &layer_values,
                   const MatrixXd &target, const ScoreFunc &score_func,
                   double step);
  MatrixXd GetGradMatrix(const MatrixXd &input, const MatrixXd &target,
                         const ScoreFunc &score_func) const;

  vector<Layer> layers_;
};
} // namespace network
