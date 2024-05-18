//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "ThresholdFunc.h"

namespace network {

ThresholdFunc::ThresholdFunc(FunctionType evaluate_0, FunctionType evaluate_1,
                             ThresholdId Id)
    : evaluate_0_(std::move(evaluate_0)), evaluate_1_(std::move(evaluate_1)),
      Id_(Id) {}

ThresholdFunc ThresholdFunc::create(ThresholdId threshold) {
  switch (threshold) {
  case ThresholdId::Sigmoid:
    return ThresholdFunc::create<ThresholdId::Sigmoid>();
  case ThresholdId::ReLu:
    return ThresholdFunc::create<ThresholdId::ReLu>();
  case ThresholdId::LeakyRelu:
    return ThresholdFunc::create<ThresholdId::LeakyRelu>();
  case ThresholdId::Default:
    return ThresholdFunc::create<ThresholdId::Default>();
  default:
    return ThresholdFunc::create<ThresholdId::Sigmoid>();
  }
}

Matrix ThresholdFunc::apply(const Matrix &layer_val) const {
  assert(evaluate_0_ && "empty apply function");
  return evaluate_0_(layer_val);
}

Matrix ThresholdFunc::derive(const Matrix &layer_val) const {
  assert(evaluate_1_ && "empty derive function");
  return evaluate_1_(layer_val);
}

bool ThresholdFunc::check_empty() { return evaluate_0_ && evaluate_1_; }

ThresholdId ThresholdFunc::GetId() const { return Id_; };
} // namespace network
