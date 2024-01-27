//
// Created by Denis Ryapolov on 22.01.2024.
//

#include "tests.h"
namespace network {
using MatrixXd = Network::MatrixXd;
using VectorXd = Network::VectorXd;

void Test::thres_constructor_test() {
  auto thres_sig = ThresholdFunc::create(ThresholdId::Sigmoid);
  auto thres_relu = ThresholdFunc::create(ThresholdId::ReLu);
}
void Test::thres_apply_test() {
  auto thres = ThresholdFunc::create(ThresholdId::Sigmoid);
  MatrixXd mat(2, 3);
  mat << 1, 2, 3, 4, 5, 6;
  thres.apply(mat);
}
void Test::thres_derive_test() {
  auto thres = ThresholdFunc::create(ThresholdId::ReLu);
  MatrixXd mat(2, 3);
  mat << 1, 2, 3, 4, 5, 6;
  thres.derive(mat);
}

void Test::score_constructor_test() {
  auto score_func_mse = ScoreFunc::create(ScoreId::MSE);
  auto score_func_mae = ScoreFunc::create(ScoreId::MAE);
  auto score_func_ent = ScoreFunc::create(ScoreId::CrossEntropy);
}

void Test::score_gradient_test() {
  auto score_func_mse = ScoreFunc::create(ScoreId::MSE);
  VectorXd in(3);
  VectorXd out(3);
  in << 2, 3, 4;
  out << 100, 200, 300;
  score_func_mse.gradient(in, out);
}

void Test::score_score_test() {
  auto score_func_mse = ScoreFunc::create(ScoreId::MSE);
  VectorXd in(3);
  VectorXd out(3);
  score_func_mse.score(in, out);
}

void Test::layer_constructor_test() {
  auto layer = Layer(ThresholdId::ReLu, 2, 3);
}
void Test::layer_grad_test1() {
  auto layer = Layer(ThresholdId::ReLu, 2, 4);
  MatrixXd grad(4, 2);
  MatrixXd applied_val(4, 2);
  layer.gradx(grad, applied_val);
}
void Test::layer_grad_test2() {
  auto layer = Layer(ThresholdId::ReLu, 2, 3);
  MatrixXd val(2, 2);
  MatrixXd applied_val(3, 2);
  MatrixXd grad(3, 2);
  layer.apply_gradA(val, grad, applied_val, 1);
}
void Test::layer_grad_test3() {
  auto layer = Layer(ThresholdId::Sigmoid, 2, 3);
  MatrixXd applied_val(3, 2);
  MatrixXd grad(3, 2);
  layer.apply_gradb(grad, applied_val, 0.5);
}

void Test::network_constructor_test() {
  auto net = Network({2, 3, 4}, {ThresholdId::ReLu, ThresholdId::Sigmoid});
}
void Test::network_train_test() {
  auto net = Network({2, 3, 4}, {ThresholdId::ReLu, ThresholdId::Sigmoid});
  MatrixXd batch(2, 3);
  MatrixXd target(4, 3);
  net.Train(batch, target, ScoreFunc::create(ScoreId::MAE), 42, 0.5);
}
void Test::network_calc_test() {
  auto net = Network({2, 3, 4}, {ThresholdId::ReLu, ThresholdId::Sigmoid});
  VectorXd batch(2);
  batch << 1, 2;
  net.Calculate(batch);
}
void Test::run_all_tests() {
  thres_constructor_test();
  thres_apply_test();
  thres_derive_test();

  score_constructor_test();
  score_score_test();
  score_gradient_test();

  layer_constructor_test();
  layer_grad_test1();
  layer_grad_test2();
  layer_grad_test3();

  network_constructor_test();
  network_train_test();
  network_calc_test();
}

} // namespace network
