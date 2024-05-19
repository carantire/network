#include "except.h"
#include "tests.h"

int main() {
  try {
    test::run_all_tests();
  } catch (...) {
    except::react();
  }
  return 0;
}
