#pragma once

#include "Layer.h"
#include "Score_Func.h"
#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>
#include <iostream>
#include <random>
#include <utility>

namespace network {

struct Values {
  using VectorXd = Eigen::VectorXd;
  template <class T> using vector = std::vector<T>;
  vector<VectorXd> in;
  vector<VectorXd> out;
};

class Network {
  template <class T> using vector = std::vector<T>;

public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  using PermutationMatrix =
      Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic>;

  Network(std::initializer_list<int> dimensions,
          std::initializer_list<Threshold_Id> threshold_id);

  VectorXd Apply(const VectorXd &start_vec);

  void TrainSGD(const MatrixXd &start_batch, const MatrixXd &reference,
                const Score_Func &score_func, double needed_accuracy,
                int max_epochs);
  void TrainBGD(const MatrixXd &start_batch, const MatrixXd &reference,
                const Score_Func &score_func, int cols_in_minibatch,
                double needed_accuracy, int max_epochs);

private:
  PermutationMatrix GetRandMat(int cols);
  VectorXd Back_Prop(const vector<Values> &values, const MatrixXd &reference,
                     const Score_Func &score_func, double step);

  VectorXd Back_Prop_SGD(const MatrixXd &start_batch, const MatrixXd &reference,
                         const Score_Func &score_func, int iter_num);

  Values Forward_Prop(const VectorXd &start_vec);
  vector<Layer> layers_;
  vector<Threshold_Id> threshold_id_;
  inline static std::minstd_rand index_generator_;
};
} // namespace network
