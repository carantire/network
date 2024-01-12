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

  Network(std::initializer_list<int> dimensions,
          std::initializer_list<Threshold_Id> threshold_id,
          Threshold_Id final_threshold_func);

  Values Forward_Prop(const VectorXd &start_vec);

  VectorXd Back_Prop(const VectorXd &start_vec, const VectorXd &reference,
                     const Score_Func &score_func, double coef);

  VectorXd Back_Prop_BGD(const VectorXd &start_vec, const VectorXd &reference,
                         const Score_Func &score_func, int iter_num);

  VectorXd Back_Prop_SGD(const MatrixXd &start_batch, const MatrixXd &reference,
                         const Score_Func &score_func, int iter_num);

private:
  vector<Layer> layers_;
  vector<Threshold_Id> threshold_id_;
  inline static std::minstd_rand index_generator;
};
} // namespace network
