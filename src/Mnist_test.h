#include "Network.h"
#include "except.h"
#include "tests.h"
#include <EigenRand/EigenRand>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include <mnist/mnist_reader.hpp>

using namespace network;
class Mnist_test {
  template <class T> using vector = std::vector<T>;
  using MatrixXd = Network::MatrixXd;
  static MatrixXd Mnist_input(const vector<vector<unsigned char>> &mat);

  static MatrixXd Mnist_output(const vector<unsigned char> &mat);

public:
  static void test();
};
