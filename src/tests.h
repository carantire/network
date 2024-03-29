#pragma once

#include "Network.h"
namespace network {
class Test {
public:
  static void thres_constructor_test();
  static void sig_apply_test();
  static void sig_derive_test();
  static void relu_apply_test();
  static void relu_derive_test();

  static void score_constructor_test();
  static void score_gradient_test();

  static void layer_constructor_test();
  static void layer_grad_test();

  static void network_constructor_test();
  static void network_train_test();

  static void run_all_tests();
};

} // namespace network
