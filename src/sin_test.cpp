//
// Created by Denis Ryapolov on 04.03.2024.
//
#include "sin_test.h"

void sin_test::test() {
  std::random_device rd;
  std::mt19937 gen(42);
  std::uniform_real_distribution<> train_dis(-M_PI / 2, M_PI + M_PI / 2);
  int size = 1000;
  MatrixXd in_values(1, size);
  MatrixXd target_values(1, size);
  for (int i = 0; i < size; ++i) {
    double randomNum = train_dis(gen);
    in_values(0, i) = randomNum;
    target_values(0, i) = sin(randomNum);
  }

  Network net(
      {1, 50, 50, 1},
      {ThresholdId::Default, ThresholdId::Sigmoid, ThresholdId::Default}, 1, 1);
  net.Train(in_values, target_values, ScoreFunc::create(ScoreId::MSE),
            network::LearningSpeedId::Linear, {0.15}, 500, 1);
  double score = 0;
  std::uniform_real_distribution<> test_dis(0, M_PI);
  for (int i = 0; i < size; ++i) {
    double randomNum = test_dis(gen);
    VectorXd test_input(1, 1);
    test_input << randomNum;
    double predict = net.Calculate(test_input)(0, 0);
    double correct = sin(randomNum);
    score += abs(predict - correct);
    //    std::cout << "Predict: " << predict << " Correct: " << correct <<
    //    '\n';
  }
  int n = 10000;
  vector<double> val(n + 1);
  for (int i = 0; i <= n; ++i){
    VectorXd test_input(1, 1);
    test_input << i * M_PI / n;
    val[i] = net.Calculate(test_input)(0, 0);
  }
  std::ofstream out_file("out.data");
  out_file.write((char *)val.data(), val.size() * sizeof(val[0]));
  std::cout << "Total score: " << score / size;
}
