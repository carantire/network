#include "Network.h"
#include <cassert>

using namespace network;

int main() {
    Threshold_Func test_func_sigmoid = Threshold_Func::create<Threshold_Id::Sigmoid>();
    assert(test_func_sigmoid.evaluate_0(-10) - 1.0 / (1.0 + std::exp(10)) < 1e-8);
    assert(test_func_sigmoid.evaluate_1(4) - std::exp(-4) / ((1.0 + std::exp(-4)) * (1.0 + std::exp(-4))) < 1e-8);
    Eigen::Vector4d test_vec(-1, 0, 1, 2);
    Eigen::Vector4d right_vec(1.0 / (1.0 + std::exp(1)), 1.0 / (1.0 + std::exp(0)), 1.0 / (1.0 + std::exp(-1)), 1.0 / (1.0 + std::exp(-2)));
    assert(test_func_sigmoid.apply(test_vec) == right_vec);
    return 0;
}
