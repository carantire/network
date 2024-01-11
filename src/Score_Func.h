#pragma once

#include <Eigen/Eigen>
#include <EigenRand/EigenRand>
#include <utility>

namespace network {

enum class Score_Id { MSE };

struct Score_Database {
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    
    template <Score_Id>
    static double score(const VectorXd &, const VectorXd &);

    template <Score_Id>
    static VectorXd gradient(const VectorXd &, const VectorXd &);

    template <>
    inline double score<Score_Id::MSE>(const VectorXd &x, const VectorXd &reference) {
        return (x - reference).dot(x - reference);
    }

    template <>
    inline VectorXd gradient<Score_Id::MSE>(const VectorXd &x, const VectorXd &reference) {
        return 2.0 * (x - reference);
    }
};

class Score_Func {
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;

    using ScoreType = std::function<double(const VectorXd &, const VectorXd &)>;
    using GradientType = std::function<VectorXd(const VectorXd &, const VectorXd &)>;

    public:
    Score_Func(ScoreType score_func, GradientType gradient_func);

    template <Score_Id Id>
    static Score_Func create() {
        return Score_Func(Score_Database::score<Id>, Score_Database::gradient<Id>);
    }

    static Score_Func create(Score_Id score);

    double score(const VectorXd &x, const VectorXd &reference) const;

    VectorXd gradient(const VectorXd &x, const VectorXd &reference) const;

    private:
    ScoreType score_func_;
    GradientType gradient_func_;
};
} // namespace network
