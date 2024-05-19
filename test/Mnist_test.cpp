#include "Network.h"
#include <chrono>
#include <mnist/mnist_reader.hpp>
#include <iostream>

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

  Network net_store({784, 256, 10}, {ThresholdId::ReLu, ThresholdId::Sigmoid},
                    2, 1. / 13.5);

  Matrix input = Mnist_input(dataset.training_images);
  Matrix target = Mnist_output(dataset.training_labels);
  auto start = std::chrono::high_resolution_clock::now();

  net_store.Train_GD(input, target, ScoreFunc::create(ScoreId::MSE),
                     LearningRateDatabase::Constant(0.08), 10, 20, 10, true);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_time = end - start;
  std::cout << "\nTrain time: " << elapsed_time.count() << "s\n";
  int correct = 0;
  std::cout << '\n';
  auto test_input = Mnist_input(dataset.test_images);
  vector<int> FP(10, 0);
  vector<int> FN(10, 0);
  vector<int> average(10, 0);
  for (int i = 0; i < dataset.test_labels.size(); ++i) {
    auto res = net_store.Calculate(test_input.col(i));
    int max_ind = 0;
    double max_val = res.maxCoeff(&max_ind);
    correct += max_ind == dataset.test_labels[i];
    if (max_ind != dataset.test_labels[i]) {
      ++FN[dataset.test_labels[i]];
      ++FP[max_ind];
    }
    ++average[dataset.test_labels[i]];
  }
  std::cout << "Correct results out of 10000: " << correct << '\n';
  std::cout << "Accuracy: "
            << 100 * double(correct) / dataset.test_labels.size() << "%\n";
  double F1 = 0;
  for (int i = 0; i <= 9; ++i) {
    double precision =
        double((average[i] - FP[i])) / (average[i] - FP[i] + FN[i]);
    double recall = double((average[i] - FP[i])) / (average[i]);
    F1 += (2 * precision * recall / (precision + recall)) *
          (double(average[i]) / dataset.test_labels.size());
  }
  std::cout << "F1: " << 100 * F1 << "%" << '\n';
}