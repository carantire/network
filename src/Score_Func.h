#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

enum class Score_Id { MSE, MAE };

struct Score_Database {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  template <Score_Id> static double score(const VectorXd &, const VectorXd &);

  template <Score_Id>
  static VectorXd gradient(const VectorXd &, const VectorXd &);

  template <>
  inline double score<Score_Id::MSE>(const VectorXd &x,
                                     const VectorXd &reference) {
    return (x - reference).dot(x - reference);
  }

  template <>
  inline VectorXd gradient<Score_Id::MSE>(const VectorXd &x,
                                          const VectorXd &reference) {
    return 2.0 * (x - reference);
  }

  template <>
  inline double score<Score_Id::MAE>(const VectorXd &x,
                                     const VectorXd &reference) {
    return (x - reference).array().abs().sum();
  }

  template <>
  inline VectorXd gradient<Score_Id::MAE>(const VectorXd &x,
                                          const VectorXd &reference) {
    return (x - reference).unaryExpr([](double el) {
      return el > 0 ? 1.0 : -1.0;
    });
  }
};

class Score_Func {
  using MatrixXd = Eigen::MatrixXd;
  using VectorXd = Eigen::VectorXd;

  using ScoreFuncType =
      std::function<double(const VectorXd &, const VectorXd &)>;
  using GradientFuncType =
      std::function<VectorXd(const VectorXd &, const VectorXd &)>;

public:
  Score_Func(ScoreFuncType score_func, GradientFuncType gradient_func);

  template <Score_Id Id> static Score_Func create() {
    return Score_Func(Score_Database::score<Id>, Score_Database::gradient<Id>);
  }

  static Score_Func create(Score_Id score);

  double score(const VectorXd &x, const VectorXd &reference) const;

  VectorXd gradient(const VectorXd &x, const VectorXd &reference) const;

private:
  ScoreFuncType score_func_;
  GradientFuncType gradient_func_;
};
} // namespace network
