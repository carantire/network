//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "ThresholdFunc.h"

namespace network {
using VectorXd = ThresholdFunc::VectorXd;
using MatrixXd = ThresholdFunc::MatrixXd;

ThresholdFunc::ThresholdFunc(FunctionType evaluate_0, FunctionType evaluate_1)
    : evaluate_0_(std::move(evaluate_0)), evaluate_1_(std::move(evaluate_1)) {}

ThresholdFunc ThresholdFunc::create(ThresholdId threshold) {
  switch (threshold) {
  case ThresholdId::Sigmoid:
    return ThresholdFunc::create<ThresholdId::Sigmoid>();
  case ThresholdId::ReLu:
    return ThresholdFunc::create<ThresholdId::ReLu>();
  case ThresholdId::SoftMax:
    return ThresholdFunc::create<ThresholdId::SoftMax>();
  default:
    return ThresholdFunc::create<ThresholdId::Sigmoid>();
  }
}

MatrixXd ThresholdFunc::apply(const MatrixXd &layer_val) const {
  assert(evaluate_0_ && "empty apply function");
  return evaluate_0_(layer_val);
}

MatrixXd ThresholdFunc::derive(const MatrixXd &layer_val) const {
  assert(evaluate_1_ && "empty derive function");
  return evaluate_1_(layer_val);
}

bool ThresholdFunc::check_empty() { return evaluate_0_ && evaluate_1_; };
} // namespace network
