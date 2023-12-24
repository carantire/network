#include <iostream>
#include <cmath>
#include <utility>
#include "Eigen/Dense"
#include "EigenRand/EigenRand"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

enum class Threshold_Id {
    Sigmoid,
    ReLu
};

struct Threshold_Database {
    template<Threshold_Id>
    static double evaluate_0(double);

    template<Threshold_Id>
    static double evaluate_1(double);

    template<>
    inline double evaluate_0<Threshold_Id::Sigmoid>(double x) {
        return 1. / (1. + std::exp(-x));
    }

    template<>
    inline double evaluate_1<Threshold_Id::Sigmoid>(double x) {
        return std::exp(-x) * evaluate_0<Threshold_Id::Sigmoid>(x) * evaluate_0<Threshold_Id::Sigmoid>(x);
    }


    template<>
    inline double evaluate_0<Threshold_Id::ReLu>(double x) {
        return x > 0 ? x : 0;
    }

    template<>
    inline double evaluate_1<Threshold_Id::ReLu>(double x) {
        return x > 0 ? 1 : 0;
    }

};

class Threshold_Func {
    using FunctionType = std::function<double(double)>;
public:
    Threshold_Func(FunctionType evaluate_0, FunctionType evaluate_1) : evaluate_0_(std::move(evaluate_0)),
                                                                       evaluate_1_(std::move(evaluate_1)) {
    }

    template<Threshold_Id Id>
    static Threshold_Func create() {
        return Threshold_Func(Threshold_Database::evaluate_0<Id>, Threshold_Database::evaluate_1<Id>);
    }

    static Threshold_Func create(Threshold_Id threshold) {
        switch (threshold) {
            case Threshold_Id::Sigmoid:
                return create<Threshold_Id::Sigmoid>();
            case Threshold_Id::ReLu:
                return create<Threshold_Id::ReLu>();
            default:
                return create<Threshold_Id::Sigmoid>();
        }
    }

    double evaluate_0(double x) const {
        return evaluate_0_(x);
    }

    double evaluate_1(double x) const {
        return evaluate_1_(x);
    }

    VectorXd apply(const VectorXd &vec) const {
        return vec.unaryExpr([this](double x) { return evaluate_0(x); });
    }

    VectorXd derive(const VectorXd &vec) const {
        return vec.unaryExpr([this](double x) { return evaluate_1(x); });

    }

private:
    FunctionType evaluate_0_;
    FunctionType evaluate_1_;
};

class Layer {


public:
    Layer(Threshold_Id sigma, int rows, int columns) : sigma_(Threshold_Func::create(sigma)),
                                                       A_(Eigen::Rand::normal<MatrixXd>(rows, columns, urng)),
                                                       b_(Eigen::Rand::normal<VectorXd>(rows, 1, urng)) {
    }

    VectorXd apply(const VectorXd &x) const { // vector of values
        return sigma_.apply(A_ * x + b_);
    }

    MatrixXd derive(const VectorXd &vec) const { // vec is a matrix of y_i = (Ax + b)_i - result of apply
        return sigma_.derive(vec).asDiagonal();
    }

    MatrixXd gradA(const VectorXd &x, const VectorXd &u, const VectorXd &vec) const { // u is a gradient vector
        return derive(vec) * u.transpose() * x.transpose();
    }

    MatrixXd gradb(const VectorXd &u, const VectorXd &vec) const {
        return derive(vec) * u.transpose();
    }

    VectorXd gradx(const VectorXd &x, const VectorXd &u, const VectorXd &vec) const {
        return (A_.transpose() * derive(vec) * u.transpose()).transpose();
    }

private:
    static Eigen::Rand::Vmt19937_64 urng;
    Threshold_Func sigma_;
    MatrixXd A_;
    VectorXd b_;
};

Eigen::Rand::Vmt19937_64 Layer::urng = 1;

enum class Score_Id {
    MSE
};

struct Score_Database {
    template<Score_Id>
    static double score(const VectorXd &, const VectorXd &);

    template<Score_Id>
    static VectorXd gradient(const VectorXd &, const VectorXd &);

    template<>
    inline double score<Score_Id::MSE>(const VectorXd &x, const VectorXd &reference) {
        return (x - reference).dot(x - reference);
    }

    template<>
    inline VectorXd gradient<Score_Id::MSE>(const VectorXd &x, const VectorXd &reference) {
        return 2 * (x - reference);
    }


};

class Score_Func {
    using ScoreType = std::function<double(const VectorXd &, const VectorXd &)>;
    using GradientType = std::function<VectorXd(const VectorXd &, const VectorXd &)>;
public:
    Score_Func(ScoreType score_func, GradientType gradient_func) : score_func_(std::move(score_func)),
                                                                   gradient_func_(std::move(gradient_func)) {}

    template<Score_Id Id>
    static Score_Func create() {
        return Score_Func(Score_Database::score<Id>, Score_Database::gradient<Id>);
    }

    static Score_Func create(Score_Id score) {
        switch (score) {
            default:
                return create<Score_Id::MSE>();
        }
    }

    double score(const VectorXd &x, const VectorXd &reference) const {
        return score_func_(x, reference);
    }

    VectorXd gradient(const VectorXd &x, const VectorXd &reference) const {
        return gradient_func_(x, reference);
    }

private:
    ScoreType score_func_;
    GradientType gradient_func_;
};

class Network {
    Network(vector<int> dimensions, vector<Threshold_Id> threshold_funcs) : dimensions_(std::move(dimensions)),
                                                                            threshold_funcs_(
                                                                                    std::move(threshold_funcs)) {
        layers.reserve(dimensions_.size() - 1);
        for (size_t i = 0; i + 1 < dimensions_.size(); ++i) {
            layers.emplace_back(Layer(threshold_funcs_[i], dimensions_[i], dimensions_[i + 1]));
        }
    }

private:
    vector<Layer> layers;
    vector<int> dimensions_;
    vector<Threshold_Id> threshold_funcs_;
};

int main() {
    Threshold_Func::create<Threshold_Id::Sigmoid>();

    std::cout << "Hello, World!" << std::endl;
    using Vector3f = Eigen::Matrix<float, 3, 1>;
    Vector3f a;
    for (auto el: a) {
        std::cout << el << '\n';
    }
    return 0;
}