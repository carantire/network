#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <cmath>

namespace network {

enum class ThresholdId { Sigmoid, ReLu };

struct ThresholdDatabase {
  template <ThresholdId> static double evaluate_0(double);

  template <ThresholdId> static double evaluate_1(double);

  template <> inline double evaluate_0<ThresholdId::Sigmoid>(double x) {
    return 1. / (1. + std::exp(-x));
  }

  template <> inline double evaluate_1<ThresholdId::Sigmoid>(double x) {
    return 1. / (std::exp(-x) + std::exp(x) + 2.);
  }

  template <> inline double evaluate_0<ThresholdId::ReLu>(double x) {
    return x * (x > 0);
  }

  template <> inline double evaluate_1<ThresholdId::ReLu>(double x) {
    return x > 0 ? 1 : 0;
  }
};

class ThresholdFunc {
  using FunctionType = std::function<double(double)>;

public:
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;
  ThresholdFunc(FunctionType evaluate_0, FunctionType evaluate_1);

  template <ThresholdId Id> static ThresholdFunc create() {
    return ThresholdFunc(ThresholdDatabase::evaluate_0<Id>,
                          ThresholdDatabase::evaluate_1<Id>);
  }

  static ThresholdFunc create(ThresholdId threshold);

  double evaluate_0(double x) const;

  double evaluate_1(double x) const;

  MatrixXd apply(const MatrixXd &layer_val) const;

  MatrixXd derive(const MatrixXd &layer_val) const;

  bool check_empty();

private:
  FunctionType evaluate_0_;
  FunctionType evaluate_1_;
};
} // namespace network
