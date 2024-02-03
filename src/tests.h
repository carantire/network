#pragma once

#include "Network.h"
namespace network {
class Test {
public:
  static void thres_constructor_test();
  static void thres_apply_test();
  static void thres_derive_test();

  static void score_constructor_test();
  static void score_gradient_test();
  static void score_score_test();

  static void layer_constructor_test();
  static void layer_grad_test1();
  static void layer_grad_test2();
  static void layer_grad_test3();

  static void network_constructor_test();
  static void network_train_test();
  static void network_calc_test();

  static void run_all_tests();
};

} // namespace network
