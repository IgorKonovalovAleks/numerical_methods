#pragma once
#include <QObject>
#include <QElapsedTimer>
#include <vector>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/math/constants/constants.hpp>

#define type_d high_precision_type

using high_precision_type = boost::multiprecision::cpp_dec_float_50;

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

class solver : public QObject {
    Q_OBJECT

    type_d (*u0y)(type_d);
    type_d (*u1y)(type_d);
    type_d (*ux0)(type_d);
    type_d (*ux1)(type_d);
    type_d (*f)(type_d x, type_d y);
    type_d (*u)(type_d x, type_d y);

    inline int is_border(int i, int j);

    void fill_right_side(std::vector<std::vector<type_d>>& v, type_d a, type_d c);
    void fill_right_side2(std::vector<std::vector<type_d>>& v, type_d a, type_d c);

    void calc_r(std::vector<std::vector<type_d>>& v);
    void calc_r_vec(std::vector<std::vector<type_d>>& v, std::vector<type_d>& res);

    void prepare(std::vector<std::vector<type_d>>& v, std::vector<std::vector<type_d>>& z, type_d a, type_d c);
    void prepare(std::vector<std::vector<type_d>>& v, type_d a, type_d c);

    void step(std::vector<std::vector<type_d>>& v, std::vector<std::vector<type_d>>& z, type_d a, type_d c, type_d& mz, type_d& acc) ;
    void step(std::vector<std::vector<type_d>>& v, type_d a, type_d c, type_d& acc);

    void step_mvr(std::vector<std::vector<type_d>>& v, std::vector<std::vector<type_d>>& z, type_d a, type_d c, type_d& mz, type_d& acc) ;
    void step_mvr(std::vector<std::vector<type_d>>& v, type_d a, type_d c, type_d& acc) ;

    void step_msi(std::vector<std::vector<type_d>>& v, std::vector<std::vector<type_d>>& z, type_d a, type_d c, type_d& mz, type_d& acc, type_d& tau, std::vector<type_d>& r);
    void step_msi(std::vector<std::vector<type_d>>& v, type_d a, type_d c, type_d& acc, type_d& tau, std::vector<type_d>& r);

    void copy(std::vector<std::vector<type_d>>& v1, std::vector<std::vector<type_d>>& z1, std::vector<std::vector<type_d>>& v2, std::vector<std::vector<type_d>>& z2);
    void copy(std::vector<std::vector<type_d>>& v1, std::vector<std::vector<type_d>>& v2);

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
    Q_INVOKABLE void solve(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps, int m_it, std::vector<std::vector<std::vector<type_d>>>& v, std::vector<std::vector<std::vector<type_d>>>& z);
    Q_INVOKABLE void solve(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps,int m_it, std::vector<std::vector<std::vector<type_d>>>& v);
signals:
    void progressUpdate(int progress, type_d accuracy, qint64 elapsed, int curr_it);
    void solveFinished();
};
