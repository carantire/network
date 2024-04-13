#include "Mnist_test.h"
#include "sin_test.h"
#include "Network.h"
#include "except.h"
#include "tests.h"
#include "Xor_test.h"
#include <EigenRand/EigenRand>
#include <cassert>
#include <cmath>
#include <exception>
#include <iostream>
#include <random>
#include <utility>

using namespace std;
int main() {
    using namespace network;
    try {
      Test::run_all_tests();
    } catch (...) {
      except::react();
    }
    Mnist_test::test();
//    sin_test::test();
//    Xor_test xor_test;
//    xor_test.test();
  return 0;
}