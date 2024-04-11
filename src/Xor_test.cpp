//
// Created by Denis Ryapolov on 11.04.2024.
//

#include "Xor_test.h"

void Xor_test::test() {
  int k = 200;
  xor_train_input_.resize(2, k);
  xor_train_output_.resize(1, k);
  for (int i = 0; i < k; ++i) {
    int x1 = rand() % 2;
    int x2 = rand() % 2;
    int y = (x1 != x2);
    xor_train_input_(0, i) = x1;
    xor_train_input_(1, i) = x2;
    xor_train_output_(0, i) = y;
  }
  Network net({2, 4, 4, 4, 1},
              {ThresholdId::ReLu, ThresholdId::Sigmoid, ThresholdId::Sigmoid,
               ThresholdId::Sigmoid},
              1, 1);
  auto score_func = ScoreFunc::create(network::ScoreId::MSE);
  net.Train(xor_train_input_, xor_train_output_, score_func, 100, 1);
  int hit_cnt = 0;
  int test_cnt = 1000;
  for (int i = 0; i < test_cnt; ++i) {
    int x1 = rand() % 1000;
    int x2 = rand() % 1000;
    MatrixXd in(2, 1);
    in << x1, x2;
    hit_cnt += (net.Calculate(in)(0, 0) == (x1 ^ x2));
  }
  std::cout << "Correct out of " << test_cnt << ": " << hit_cnt;
}