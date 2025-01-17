#pragma once

#include "ThresholdFunc.h"
#include "utils.h"
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

class Layer {
public:
  Layer(ThresholdId id, Index in_size, Index out_size, RandGen &rng,
        double normalize);

  Matrix apply_linear(const Matrix &values) const;

  Matrix apply_threshold(const Matrix &value) const;

  Matrix gradx(const Matrix &grad, const Matrix &applied_values) const;

  void apply_gradA(const Matrix &values, const Matrix &grad,
                   const Matrix &applied_values, double step);

  void apply_gradb(const Matrix &grad, const Matrix &applied_values,
                   double step);

  void WriteParams(std::ofstream &out) const;

  static Layer ReadParams(std::ifstream &in);

  Index Get_Input_Dim() const;

  Matrix Get_Mat() const;

private:
  Layer(Matrix&& A, Vector&& b, ThresholdId Id)
      : A_(A), b_(b),
        ThresholdFunc_(ThresholdFunc::create(Id)){};

  static Matrix getNormal(Index rows, Index columns, RandGen &rng,
                          double normalize);

  Matrix derive(const Matrix &applied_values) const;

  Matrix derive_mat(const Matrix &applied_values, const Matrix &grad) const;

  ThresholdFunc ThresholdFunc_;

  Matrix A_;
  Vector b_;
};
} // namespace network
