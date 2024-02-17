#include "Mnist_test.h"
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

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;
int main() {
    using namespace network;
    try {
      Test::run_all_tests();
    } catch (...) {
      except::react();
    }
    Mnist_test::test();
  return 0;
}