#include "Network.h"
#include <algorithm>
#include <mnist/mnist_reader.hpp>

using namespace network;
using std::vector;

Matrix Mnist_input(const vector<vector<unsigned char>> &mat) {
  Matrix res(mat[0].size(), mat.size());
  for (int i = 0; i < mat.size(); ++i) {
    for (int j = 0; j < mat[0].size(); ++j) {
      res(j, i) = double(mat[i][j]) / 255.;
    }
  }
  return res;
}

Matrix Mnist_output(const vector<unsigned char> &mat) {
  Matrix res(10, mat.size());
  for (int i = 0; i < mat.size(); ++i) {
    for (int j = 0; j < 10; ++j) {
      res(j, i) = j == mat[i] ? 1 : 0;
    }
  }
  return res;
}

int main() {
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
          MNIST_DATA_LOCATION);

  Network net_store({784, 512, 10}, {ThresholdId::ReLu, ThresholdId::Sigmoid}, 1,
              1. / 12);

  Matrix input = Mnist_input(dataset.training_images);
  Matrix target = Mnist_output(dataset.training_labels);
  net_store.Train_GD(input, target, ScoreFunc::create(ScoreId::MSE),
               LearningRateDatabase::Constant(0.1), 10, 30);
  net_store.StoreModel("out.data");
  Network net = Network::LoadModel("out.data");

  int correct = 0;
  std::cout << '\n';
  auto test_input = Mnist_input(dataset.test_images);
  vector<int> mistake(10, 0);
  vector<int> average(10, 0);
  for (int i = 0; i < 10000; ++i) {
    auto res = net.Calculate(test_input.col(i));
    int max_ind = 0;
    double max_val = res.maxCoeff(&max_ind);
    correct += max_ind == dataset.test_labels[i];
    if (max_ind != dataset.test_labels[i]) {
      ++mistake[dataset.test_labels[i]];
    }
    ++average[dataset.test_labels[i]];
  }
  std::cout << "Correct results out of 10000: " << correct << '\n';
  for (int i = 0; i < 10; ++i) {
    std::cout << "mistakes while predicting " << i << ": "
              << 100 * double(mistake[i]) / average[i] << "%" << '\n';
  }
}