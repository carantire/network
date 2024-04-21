#include "tests.h"
#include "Network.h"

using namespace network;
using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace {
void thres_constructor_test() {
  auto thres_sig = ThresholdFunc::create(ThresholdId::Sigmoid);
  auto thres_relu = ThresholdFunc::create(ThresholdId::ReLu);
}

void sig_apply_test() {
  auto thres = ThresholdFunc::create(ThresholdId::Sigmoid);
  MatrixXd mat(2, 3);
  mat << 1, 2, 3, 4, 5, 6;
  mat = thres.apply(mat);
  MatrixXd ans(2, 3);
  ans << 0.73105, 0.880797, 0.952574, 0.98201379, 0.993307149, 0.9975273;
  assert(abs((mat - ans).array().maxCoeff()) < 1e-5);
}

void sig_derive_test() {
  auto thres = ThresholdFunc::create(ThresholdId::Sigmoid);
  MatrixXd mat(2, 3);
  mat << -10, -1, 0, 1, 2, 10;
  mat = thres.derive(mat);
  MatrixXd ans(2, 3);
  ans << 4.539580, 0.1966119, 0.25, 0.1966119, 0.10499358, 4.5395807;
  assert(abs((mat - ans).array().maxCoeff()) < 1e-5);
}

void relu_apply_test() {
  auto thres = ThresholdFunc::create(ThresholdId::ReLu);
  MatrixXd mat(2, 3);
  mat << -10, -1, 0, 1, 2, 10;
  mat = thres.apply(mat);
  MatrixXd ans(2, 3);
  ans << 0, 0, 0, 1, 2, 10;
  assert(abs((mat - ans).array().maxCoeff()) < 1e-5);
}

void relu_derive_test() {
  auto thres = ThresholdFunc::create(ThresholdId::ReLu);
  MatrixXd mat(2, 3);
  mat << -10, -1, 0, 1, 2, 10;
  mat = thres.derive(mat);
  MatrixXd ans(2, 3);
  ans << 0, 0, 0, 1, 1, 1;
  assert(abs((mat - ans).array().maxCoeff()) < 1e-5);
}

void score_constructor_test() {
  auto score_func_mse = ScoreFunc::create(ScoreId::MSE);
  auto score_func_mae = ScoreFunc::create(ScoreId::MAE);
  auto score_func_ent = ScoreFunc::create(ScoreId::CrossEntropy);
}

void score_gradient_test() {
  auto score_func_mse = ScoreFunc::create(ScoreId::MSE);
  VectorXd in(3);
  VectorXd out(3);
  in << 2, 3, 4;
  out << 100, 200, 300;
  VectorXd grad = score_func_mse.gradient(in, out);
  VectorXd ans(3);
  ans << 2 * (-98), 2 * (3 - 200), 2 * (4 - 300);
  assert(abs((grad - ans).array().maxCoeff()) < 1e-5);
}

void layer_constructor_test() {
  auto layer = Layer(ThresholdId::ReLu, 2, 3, 1, 1);
}

void layer_grad_test() {
  auto layer = Layer(ThresholdId::ReLu, 2, 2, 1, 1);
  MatrixXd A(2, 2);
  A << 1, 1, 1, 1;
  MatrixXd grad(2, 2);
  grad << -1, 0, -1, 0;
  MatrixXd applied_val(2, 2);
  applied_val << 2, 1, 1, 2;
  auto new_grad = layer.gradx(grad, applied_val);
  MatrixXd ans(2, 2);
  ans << -1, 0, -1, 0;
  assert(
      abs((new_grad - layer.Get_Mat().transpose() * ans).array().maxCoeff()) <
      1e-5);
}

void network_constructor_test() {
  auto net =
      Network({2, 3, 4}, {ThresholdId::ReLu, ThresholdId::Sigmoid}, 1, 1);
}

void network_train_test() {
  auto net = Network({2, 3, 4, 5, 4},
                     {ThresholdId::ReLu, ThresholdId::Sigmoid,
                      ThresholdId::Sigmoid, ThresholdId::LeakyRelu},
                     1, 1);
  MatrixXd batch(2, 3);
  MatrixXd target(4, 3);
  net.Train(batch, target, ScoreFunc::create(ScoreId::CrossEntropy),
            network::LearningRateDatabase::Constant(1), 1, 1);
}
}

void test::run_all_tests() {
  thres_constructor_test();
  sig_apply_test();
  sig_derive_test();
  relu_apply_test();
  relu_derive_test();

  score_constructor_test();
  score_gradient_test();

  layer_constructor_test();
  layer_grad_test();

  network_constructor_test();
  network_train_test();
}