#include "Network.h"
#include <cassert>

using namespace network;
using namespace std;
using MatrixXd = Network::MatrixXd;
using VectorXd = Network::VectorXd;


int main() {
  Threshold_Func test_func_sigmoid =
      Threshold_Func::create<Threshold_Id::Sigmoid>();
  assert(test_func_sigmoid.evaluate_0(-10) - 1.0 / (1.0 + std::exp(10)) < 1e-8);
  assert(test_func_sigmoid.evaluate_1(4) -
             std::exp(-4) / ((1.0 + std::exp(-4)) * (1.0 + std::exp(-4))) <
         1e-8);
  Eigen::Vector4d test_vec(-1.0, 0.0, 1.0, 2);
  Eigen::Vector4d right_vec(
      1.0 / (1.0 + std::exp(1)), 1.0 / (1.0 + std::exp(0)),
      1.0 / (1.0 + std::exp(-1)), 1.0 / (1.0 + std::exp(-2)));
  assert(test_func_sigmoid.apply(test_vec) == right_vec);
  Score_Func test_score = Score_Func::create(Score_Id::MSE);
  Eigen::Vector4d test_vec_2(-1.0, 1.0, 1.0, 2);
  assert(test_score.score(test_vec, test_vec_2) == 1);
  Eigen::Vector4d diff_vec(0.0, -2, 0.0, 0.0);
  assert(test_score.gradient(test_vec, test_vec_2) == diff_vec);
  Layer test_layer(Threshold_Id::ReLu, 2, 2);
//  Network test_net({4, 2, 2}, {Threshold_Id::Sigmoid, Threshold_Id::ReLu});
//  MatrixXd in(4, 2);
//  in << 1, 2, 3, 4, 5, 6, 7, 8;
//  auto test_val = test_net.Forward_Prop(in.col(0));
//  cout << test_val.in[1].rows();
//  MatrixXd reference(2, 2);
//  reference << 2, 2, 3, 3;
//  vector<Values> v = {test_val};
//  cout << VectorXd::Zero(v[0].in.front().rows());
//  cout << v[0].in.size();
//  cout << test_net.Back_Prop(v, reference.col(0), test_score, 1);
//  test_net.TrainSGD(in, reference, test_score, 0.03, 200);
//  test_net.TrainBGD(in, reference, test_score, 2, 0.05, 100);
  return 0;
}
