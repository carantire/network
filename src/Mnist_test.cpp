#include "Mnist_test.h"
#include <algorithm>


Mnist_test::MatrixXd Mnist_test::Mnist_input(const vector<vector<unsigned char>> &mat) {
  MatrixXd res(mat[0].size(), mat.size());
  for (int i = 0; i < mat.size(); ++i) {
    for (int j = 0; j < mat[0].size(); ++j) {
      res(j, i) = double(mat[i][j]) / 255.;
    }
  }
  return res;
}

Mnist_test::MatrixXd Mnist_test::Mnist_output(const vector<unsigned char> &mat) {
  MatrixXd res(10, mat.size());
  for (int i = 0; i < mat.size(); ++i) {
    for (int j = 0; j < 10; ++j) {
      res(j, i) = j == mat[i] ? 1 : 0;
    }
  }
  return res;
}

void Mnist_test::test() {
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
          MNIST_DATA_LOCATION);

  Network net({784, 256, 10}, {ThresholdId::ReLu, ThresholdId::Sigmoid}, 1,
              1. / 12);
  MatrixXd input = Mnist_input(dataset.training_images);
  MatrixXd target = Mnist_output(dataset.training_labels);
  net.Train(input, target, ScoreFunc::create(ScoreId::MSE),
            network::LearningSpeedId::Linear, {0.15}, 10, 30);

  int correct = 0;
  std::cout << '\n';
  auto test_input = Mnist_input(dataset.test_images);
  for (int i = 0; i < 10000; ++i) {
    auto res = net.Calculate(test_input.col(i));
    int max_ind = 0;
    double max_val = res.maxCoeff(&max_ind);
    correct += max_ind == dataset.test_labels[i];
  }
  std::cout << "Correct results out of 10000: " << correct << '\n';
}