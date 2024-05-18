#pragma once
#include <functional>
#include <cmath>

namespace network {
using LearningRate = std::function<double(int)>;

namespace LearningRateDatabase {
LearningRate Exponent(double c1, double c2);
LearningRate Linear(double c1);
LearningRate Constant(double c1);
} // namespace LearningRateDatabase
} // namespace network