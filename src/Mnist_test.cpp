#include "Mnist_test.h"
#include <algorithm>

using MatrixXd = Eigen::MatrixXd;
using VectorXd = Eigen::VectorXd;

MatrixXd Mnist_test::MatConstructor(const vector<vector<unsigned char>> &mat,
                                    int start_ind, int batch_size) {
  MatrixXd res(mat[0].size(), batch_size);
  for (int i = start_ind; i < start_ind + batch_size; ++i) {
    for (int j = 0; j < mat[0].size(); ++j) {
      res(j, i - start_ind) = double(mat[i][j]) / 255.;
    }
  }
  return res;
}

MatrixXd Mnist_test::OutConstructor(const vector<unsigned char> &mat,
                                    int start_ind, int batch_size) {
  MatrixXd res(10, batch_size);
  for (int i = start_ind; i < start_ind + batch_size; ++i) {
    for (int j = 0; j < 10; ++j) {
      res(j, i - start_ind) = j == mat[i] ? 1 : 0;
    }
  }
  return res;
}

void Mnist_test::test() {
  // MNIST_DATA_LOCATION set by MNIST cmake config
  std::cout << "MNIST data directory: " << MNIST_DATA_LOCATION << std::endl;

  // Load MNIST data
  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset =
      mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(
          MNIST_DATA_LOCATION);
  std::cout << "Nbr of training images = " << dataset.training_images.size()
            << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size()
            << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size()
            << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size()
            << std::endl;

  Network net({784, 256, 10},
              {ThresholdId::ReLu, ThresholdId::Sigmoid});

  for (int epoch = 0; epoch < 10; ++epoch) {
    std::cout << "Epoch num: " << epoch << '\n';
    std::cout << '\n';
    for (int k = 0; k < 2000; ++k) {
//      if (k % 100 == 0) {
//        std::cout << k << " vectors trained" << '\n';
//      }
      int batch_size = 30;
      int start_ind = k * batch_size;
      MatrixXd input =
          MatConstructor(dataset.training_images, start_ind, batch_size);
      MatrixXd target =
          OutConstructor(dataset.training_labels, start_ind, batch_size);
      net.Train(input, target, ScoreFunc::create(ScoreId::MSE), 1,
                0.15/(1 + epoch));
    }

    int correct = 0;
    std::cout << '\n';
    for (int i = 0; i < 10000; ++i) {
      if (i % 1000 == 0) {
        std::cout << i << " correct out of " << correct << '\n';
      }
      auto in = MatConstructor(dataset.test_images, i, 1);
      auto res = net.Calculate(in);
      int max_ind = 0;
      double max_val = res.maxCoeff(&max_ind);
      correct += max_ind == dataset.test_labels[i];
    }
    std::cout << "Correct results out of 10000: " << correct << '\n';

//    int correct = 0;
//    std::cout << '\n';
//    for (int i = 0; i < 60000; ++i) {
//      if (i % 10000 == 0) {
//        std::cout << i << " correct out of " << correct << '\n';
//      }
//      auto in = MatConstructor(dataset.training_images, i, 1);
//      auto res = net.Calculate(in);
//      int max_ind;
//      double max_val = res.maxCoeff(&max_ind);
//      correct += max_ind == dataset.training_labels[i];
//    }
//    std::cout << "Correct results out of 60000: " << correct << '\n';
  }
}