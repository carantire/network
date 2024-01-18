//
// Created by Denis Ryapolov on 06.01.2024.
//

#include "Threshold_Func.h"

namespace network {
using VectorXd = Threshold_Func::VectorXd;
using MatrixXd = Threshold_Func::MatrixXd;

Threshold_Func::Threshold_Func(FunctionType evaluate_0, FunctionType evaluate_1)
    : evaluate_0_(std::move(evaluate_0)), evaluate_1_(std::move(evaluate_1)) {}

Threshold_Func Threshold_Func::create(Threshold_Id threshold) {
  switch (threshold) {
  case Threshold_Id::Sigmoid:
    return Threshold_Func::create<Threshold_Id::Sigmoid>();
  case Threshold_Id::ReLu:
    return Threshold_Func::create<Threshold_Id::ReLu>();
  default:
    return Threshold_Func::create<Threshold_Id::Sigmoid>();
  }
}

double Threshold_Func::evaluate_0(double x) const {
  assert(evaluate_0_ && "Empty evaluate_0 method!");
  return evaluate_0_(x);
}

double Threshold_Func::evaluate_1(double x) const {
  assert(evaluate_1_ && "Empty evaluate_1 method!");
  return evaluate_1_(x);
}

VectorXd Threshold_Func::apply(const VectorXd &vec) const {
  return vec.unaryExpr([this](double x) { return evaluate_0(x); });
}

VectorXd Threshold_Func::derive(const VectorXd &vec) const {
  return vec.unaryExpr([this](double x) { return evaluate_1(x); });
}
} // namespace network
