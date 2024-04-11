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
using std::vector;
using MatrixXd = Eigen::MatrixXd;
class Xor_test {
  MatrixXd xor_train_input_;
  MatrixXd xor_train_output_;
public:
  void test();
};
