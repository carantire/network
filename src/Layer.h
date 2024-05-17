#pragma once

#include "ThresholdFunc.h"
#include "utils.h"
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

class Layer {
public:

  Layer(ThresholdId id, Index in_size, Index out_size, int seed,
        double normalize);

  Matrix apply_linear(const Matrix &values) const;

  Matrix apply_threshold(const Matrix &value) const;

  Matrix derive(const Matrix &applied_values) const;

  Matrix derive_mat(const Matrix &applied_values, const Matrix &grad) const;

  Matrix gradx(const Matrix &grad, const Matrix &applied_values) const;

  void apply_gradA(const Matrix &values, const Matrix &grad,
                   const Matrix &applied_values, double step);

  void apply_gradb(const Matrix &grad, const Matrix &applied_values,
                   double step);

  Index Get_Input_Dim() const;

  Matrix Get_Mat() const;

private:
  static Matrix getNormal(Index rows, Index columns, int seed,
                          double normalize);

  ThresholdFunc ThresholdFunc_;

  Matrix A_;
  Vector b_;
};
} // namespace network
