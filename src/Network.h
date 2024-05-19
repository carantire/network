#pragma once

#include "Layer.h"
#include "LearningRate.h"
#include "ScoreFunc.h"
#include "iostream"
#include <Eigen/Eigen>
#include <cmath>
#include <filesystem>

namespace network {

struct LayerValue {
  Matrix in;
  Matrix out;
};

class Network {
public:
  Network(std::initializer_list<int> dimensions,
          std::initializer_list<ThresholdId> threshold_id, int seed,
          double normalize);

  Vector Calculate(const Vector &start_vec) const;

  void Train_GD(Matrix input, Matrix target, const ScoreFunc &score_func,
                const LearningRate &learning_rate, int epoch_num,
                int batch_size, int seed, bool debug = false);
  void Train_SGD(Matrix input, Matrix target, const ScoreFunc &score_func,
                 const LearningRate &learning_rate, int epoch_num,
                 int sample_size, int seed, bool debug = false);

  void StoreModel(const std::filesystem::path &path);

  static Network LoadModel(const std::filesystem::path &path);

private:
  Network(std::vector<Layer> &&layers);
  std::vector<Layer> layers_;

  std::vector<LayerValue> Forward_Prop(const Matrix &start_vec) const;
  void Back_Prop(const std::vector<LayerValue> &layer_values,
                 const Matrix &target, const ScoreFunc &score_func,
                 double step);
  Matrix GetGradMatrix(const Matrix &input, const Matrix &target,
                       const ScoreFunc &score_func) const;
  void ShuffleData(Matrix &input, Matrix &target, RandGen &rng);
};
} // namespace network
