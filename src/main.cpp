#include "Network.h"
#include <cassert>

using namespace network;

int main() {
    Threshold_Func test_func_sigmoid = Threshold_Func::create<Threshold_Id::Sigmoid>();
    assert(test_func_sigmoid.evaluate_0(-10) - 1.0 / (1.0 + std::exp(10)) < 1e-8);
    assert(test_func_sigmoid.evaluate_1(4) - std::exp(-4) / ((1.0 + std::exp(-4)) * (1.0 + std::exp(-4))) < 1e-8);
    Eigen::Vector4d test_vec(-1.0, 0.0, 1.0, 2);
    Eigen::Vector4d right_vec(1.0 / (1.0 + std::exp(1)), 1.0 / (1.0 + std::exp(0)), 1.0 / (1.0 + std::exp(-1)), 1.0 / (1.0 + std::exp(-2)));
    assert(test_func_sigmoid.apply(test_vec) == right_vec);
    Score_Func test_score = Score_Func::create(Score_Id::MSE);
    Eigen::Vector4d test_vec_2(-1.0, 1.0, 1.0, 2);
    assert(test_score.score(test_vec, test_vec_2) == 1);
    Eigen::Vector4d diff_vec(0.0, -2, 0.0 ,0.0);
    assert(test_score.gradient(test_vec, test_vec_2) == diff_vec);
    Layer test_layer(Threshold_Id::ReLu, 2, 2);
    Network test_net({4, 2, 2}, {Threshold_Id::Sigmoid, Threshold_Id::ReLu});
    auto test_val = test_net.Forward_Prop(test_vec);
    Eigen::VectorXd a(2);
    a << 1, 2;
    auto train = test_net.Back_Prop(test_vec, a, test_score, 0.5);
    std::cout << "Hello";
    return 0;
}
