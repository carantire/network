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

using namespace network;

class Xor_test {
  using MatrixXd = Eigen::MatrixXd;
  template <class T> using vector = std::vector<T>;
  MatrixXd xor_train_input_;
  MatrixXd xor_train_output_;

public:
  void test();
};
