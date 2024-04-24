#include "LearningRate.h"

using network::LearningRate;
namespace network::LearningRateDatabase {
LearningRate Exponent(double c1, double c2) {
  return [=](int epoch) { return c1 * exp(-epoch * c2); };
}
LearningRate Linear(double c1) {
  return [=](int epoch) { return c1 / epoch; };
}
LearningRate Constant(double c1) {
  return [=](int epoch) { return c1; };
}
} // namespace network::LearningRateDatabase