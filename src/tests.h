#pragma once

#include "Network.h"
namespace network {
class Test {
public:
  void thres_constructor_test();
  void thres_apply_test();
  void thres_derive_test();

  void score_constructor_test();
  void score_gradient_test();
  void score_score_test();

  void layer_constructor_test();
  void layer_grad_test1();
  void layer_grad_test2();
  void layer_grad_test3();

  void network_constructor_test();
  void network_train_test();
  void network_calc_test();

  void run_all_tests();
};

} // namespace network
