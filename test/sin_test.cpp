#include "Network.h"
#include <Eigen/Eigen>
#include <cmath>
#include <random>
#include <vector>

int main() {
  using namespace network;

  using Eigen::MatrixXd;
  using Eigen::VectorXd;

  std::random_device rd;
  std::mt19937 gen(4);
  std::uniform_real_distribution<> train_dis(-M_PI / 2, M_PI + M_PI / 2);
  int size = 10000;
  MatrixXd in_values(1, size);
  MatrixXd target_values(1, size);
  for (int i = 0; i < size; ++i) {
    double randomNum = train_dis(gen);
    in_values(0, i) = randomNum;
    target_values(0, i) = sin(randomNum);
  }

  Network net(
      {1, 50, 50, 1},
      {ThresholdId::Sigmoid, ThresholdId::Sigmoid, ThresholdId::Sigmoid}, 1, 1);
  net.Train(in_values, target_values, ScoreFunc::create(ScoreId::MAE),
            network::LearningRateDatabase::Constant(0.005), 100, 1);
  double score = 0;
  std::uniform_real_distribution<> test_dis(0, M_PI);
  for (int i = 0; i < size; ++i) {
    double randomNum = test_dis(gen);
    VectorXd test_input(1, 1);
    test_input << randomNum;
    double predict = net.Calculate(test_input)(0, 0);
    double correct = sin(randomNum);
    score += abs(predict - correct);
  }
  int n = 10000;
  std::vector<double> val(n + 1);
  for (int i = 0; i <= n; ++i) {
    VectorXd test_input(1, 1);
    test_input << i * M_PI / n;
    val[i] = net.Calculate(test_input)(0, 0);
  }
  std::ofstream out_file("out.data");
  out_file.write((char *)val.data(), val.size() * sizeof(val[0]));
  std::cout << "Total score: " << score / size;
}
