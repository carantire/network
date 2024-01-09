#pragma once

#include "Layer.h"
#include "Threshold_Func.h"
#include <Eigen/Dense>
#include <EigenRand/EigenRand/EigenRand>
#include <utility>

namespace network {

enum class Score_Id { MSE };

struct Score_Database {
    template <Score_Id>
    static double score(const Eigen::VectorXd &, const Eigen::VectorXd &);

    template <Score_Id>
    static Eigen::VectorXd gradient(const Eigen::VectorXd &, const Eigen::VectorXd &);

    template <>
    inline double score<Score_Id::MSE>(const Eigen::VectorXd &x, const Eigen::VectorXd &reference) {
        return (x - reference).dot(x - reference);
    }

    template <>
    inline Eigen::VectorXd gradient<Score_Id::MSE>(const Eigen::VectorXd &x, const Eigen::VectorXd &reference) {
        return 2 * (x - reference);
    }
};

class Score_Func {
    using ScoreType = std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)>;
    using GradientType = std::function<Eigen::VectorXd(const Eigen::VectorXd &, const Eigen::VectorXd &)>;

    public:
    Score_Func(ScoreType score_func, GradientType gradient_func);

    template <Score_Id Id>
    static Score_Func create() {
        return Score_Func(Score_Database::score<Id>, Score_Database::gradient<Id>);
    }

    static Score_Func create(Score_Id score);

    double score(const Eigen::VectorXd &x, const Eigen::VectorXd &reference) const;

    Eigen::VectorXd gradient(const Eigen::VectorXd &x, const Eigen::VectorXd &reference) const;

    private:
    ScoreType score_func_;
    GradientType gradient_func_;
};
} // namespace network