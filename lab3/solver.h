#pragma once
#include <QObject>
#include <QElapsedTimer>
#include <vector>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/float128.hpp>
#include <boost/math/constants/constants.hpp>

using high_precision_type = boost::multiprecision::cpp_dec_float_50;
typedef high_precision_type type_d;

// boost::multiprecision::float128;

namespace u_test {
type_d u(type_d x, type_d y);

type_d u0y(type_d y);

type_d u1y(type_d y);

type_d ux0(type_d x);

type_d ux1(type_d x);

type_d f(type_d x, type_d y);
}

namespace u_main {
type_d u(type_d x, type_d y);

type_d u0y(type_d y);

type_d u1y(type_d y);

type_d ux0(type_d x);

type_d ux1(type_d x);

type_d f(type_d x, type_d y);
}

enum Functions {
    test, tmain
};

enum Methods {
    zeidel, mvr, msi
};

class Matrix: std::vector<type_d> {
    //std::vector<type_d> data;
    size_t size1, size2;
public:
    Matrix() {
        size1 = 0;
        size2 = 0;
    }
    Matrix(size_t size1, size_t size2) {
        this->size1 = size1;
        this->size2 = size2;
        ((std::vector<type_d>*)this)->reserve(size1 * size2);
    }
    void resize(size_t size1, size_t size2) {
        this->size1 = size1;
        this->size2 = size2;
        ((std::vector<type_d>*)this)->reserve(size1 * size2);
    }
    inline type_d& operator()(size_t i, size_t j) {
        return (*this)[i*size2 + j];
    }
};

class solver : public QObject {
    Q_OBJECT

    type_d (*u0y)(type_d);
    type_d (*u1y)(type_d);
    type_d (*ux0)(type_d);
    type_d (*ux1)(type_d);
    type_d (*f)(type_d x, type_d y);
    type_d (*u)(type_d x, type_d y);

    inline int is_border(int i, int j);

    void fill_right_side(Matrix& v, type_d a, type_d c);
    void fill_right_side2(Matrix& v, type_d a, type_d c);

    void calc_r_vec(std::vector<std::vector<type_d>>& v, std::vector<type_d>& res);
    void calc_r(Matrix& v);

    void prepare(Matrix& v, Matrix& z, type_d a, type_d c);
    void prepare(Matrix& v, type_d a, type_d c);

    void step(Matrix& v, Matrix& z, type_d a, type_d c, type_d& mz, type_d& acc) ;
    void step(Matrix& v, type_d a, type_d c, type_d& acc) ;

    void step_mvr(Matrix& v, Matrix& z, type_d a, type_d c, type_d& mz, type_d& acc) ;
    void step_mvr(Matrix& v, type_d a, type_d c, type_d& acc) ;

    void step_msi(std::vector<std::vector<type_d>>& v, std::vector<std::vector<type_d>>& z, type_d a, type_d c, type_d& mz, type_d& acc, type_d& tau, std::vector<type_d>& r);
    void step_msi(std::vector<std::vector<type_d>>& v, type_d a, type_d c, type_d& acc, type_d& tau, std::vector<type_d>& r);

    void copy(Matrix& v1, Matrix& z1, Matrix& v2, Matrix& z2);
    void copy(Matrix& v1, Matrix& v2);

public:
    int N, M, max_it, it, task;
    int interval = 1000;
    int meth = Methods::zeidel;
    type_d x0, X, y0, Y, k, h, hor, ver, A, max_z, achieved_accuracy, max_r, epsilon, max_x, max_y;
    bool valid = 0;
    type_d w = type_d(1.5);
    qint64 duration;
    QElapsedTimer timer;

    std::vector<int> iter;
    std::vector<type_d> right_side;
    std::vector<type_d> MAX_Z;
    std::vector<type_d> MAX_R;
    std::vector<type_d> ACCURACY;

    void constructor();
    void constructor(int function);
public slots:
    Q_INVOKABLE void solve(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps, int m_it,
                           Matrix & v, std::vector<Matrix>& vHistory,
                           Matrix & z, std::vector<Matrix>& zHistory);
    Q_INVOKABLE void solve(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps,int m_it,
                           Matrix & v, std::vector<Matrix>& vHistory);
signals:
    void progressUpdate(int progress, type_d accuracy, qint64 elapsed, int curr_it);
    void solveFinished();
};
