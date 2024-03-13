//
// Created by Denis Ryapolov on 04.03.2024.
//
#include "sin_test.h"

void sin_test::test() {
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_real_distribution<> dis(0, M_PI + 0.01);
  int size = 500;
  vector<double> in_values(size);
  vector<double> target_values(size);
  for (int i = 0; i < size; ++i) {
    double randomNum = dis(gen);
    in_values[i] = randomNum;
    target_values[i] = sin(randomNum);
  }

  Network net({1, 50, 50, 1}, {ThresholdId::Sigmoid, ThresholdId::Sigmoid,
                               ThresholdId::Sigmoid});
  for (int epoch = 0; epoch < 10; ++epoch) {
    for (int k = 0; k < size; ++k) {
      MatrixXd input(1, 1);
      MatrixXd target(1, 1);
      input << in_values[k];
      target << target_values[k];
      net.Train(input, target, ScoreFunc::create(ScoreId::MSE), 1, 1.24);
    }
  }
  double score = 0;
  for (int i = 0; i < size; ++i) {
    double randomNum = dis(gen);
    VectorXd test_input(1, 1);
    test_input << randomNum;
    score += abs(net.Calculate(test_input)(0, 0) - sin(randomNum));
    test_input << in_values[i];
    std::cout << net.Calculate(test_input) << " " << sin(randomNum) << '\n';
  }

  std::cout << score / size;
}