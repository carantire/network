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

int main() {
  using namespace network;
  try {
    Test::run_all_tests();
    return 0;
  } catch (...) {
    except::react();
  }
}
