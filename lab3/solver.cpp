#include "solver.h"
#include "omp.h"
#include <iostream>
#define type_d double

type_d u_test::u(type_d x, type_d y) {
    return exp(sin(boost::math::constants::pi<type_d>() * x * y) * sin(boost::math::constants::pi<type_d>() * x * y));
}

type_d u_test::u0y(type_d y) {
    return type_d(1);
}

type_d u_test::u1y(type_d y) {
    return exp(sin(boost::math::constants::pi<type_d>() * y) * sin(boost::math::constants::pi<type_d>() * y));
}

type_d u_test::ux0(type_d x) {
    return type_d(1);
}

type_d u_test::ux1(type_d x) {
    return exp(sin(boost::math::constants::pi<type_d>() * x) * sin(boost::math::constants::pi<type_d>() * x));
}

type_d u_test::f(type_d x, type_d y) {
    return u(x, y) * boost::math::constants::pi<type_d>() * boost::math::constants::pi<type_d>() * (x * x + y * y) * (- type_d(1) - type_d(4) * cos(type_d(2) * boost::math::constants::pi<type_d>() * x * y) + cos(type_d(4) * boost::math::constants::pi<type_d>() * x * y)) / type_d(2);
}

type_d u_main::u0y(type_d y) {
    return sin(boost::math::constants::pi<type_d>() * y);
}

type_d u_main::u1y(type_d y) {
    return sin(boost::math::constants::pi<type_d>() * y);
}

type_d u_main::ux0(type_d x) {
    return x - x * x;
}

type_d u_main::ux1(type_d x) {
    return x - x * x;
}

type_d u_main::f(type_d x, type_d y) {
    return sin(boost::math::constants::pi<type_d>() * x * y) * sin(boost::math::constants::pi<type_d>() * x * y);
}

void solver::constructor() {
    k = type_d(0);
    h = type_d(0);
    N = 0;
    M = 0;
    x0 = type_d(0);
    X = type_d(0);
    y0 = type_d(0);
    Y = type_d(0);
    task = Functions::test;

    u0y = u_test::u0y;
    u1y = u_test::u1y;
    ux0 = u_test::ux0;
    ux1 = u_test::ux1;
    u = u_test::u;
    f = u_test::f;
}

void solver::constructor(int function) {
    k = type_d(0);
    h = type_d(0);
    N = 0;
    M = 0;
    x0 = type_d(0);
    X = type_d(0);
    y0 = type_d(0);
    Y = type_d(0);

    if (function == Functions::test) {
        u0y = u_test::u0y;
        u1y = u_test::u1y;
        ux0 = u_test::ux0;
        ux1 = u_test::ux1;
        u = u_test::u;
        f = u_test::f;
        task = Functions::test;
    }
    else if (function == Functions::tmain) {
        u0y = u_main::u0y;
        u1y = u_main::u1y;
        ux0 = u_main::ux0;
        ux1 = u_main::ux1;
        f = u_main::f;
        task = Functions::tmain;
    }
}


void solver::prepare(Matrix& v, Matrix& z, type_d a, type_d c) {
    v.resize(N + 1, M + 1);
    z.resize(N + 1, M + 1);
    buf.resize(N + 1, M + 1);

    for (int i = 0; i < N + 1; i++) {
        v(i, 0) = ux0(a + h * i);
        v(i, M) = ux1(a + h * i);
    }

    for (int i = 1; i < M; i++) {
        v(0, i) = u0y(c + k * i);
        v(M, i) = u1y(c + k * i);
    }

    for (int i = 1; i < M; i++) {
      buf(0, i) = v(0, i);
      buf(M, i) = v(M, i);
    }

    for (int i = 1; i < N; i++)
        for (int j = 1; j < M; j++)
            v(i, j) = ux0(a + h * i) + k * j * (ux1(a+ h * i) - ux0(a + h * i));

    for (int i = 0; i < N + 1; i++)
        for (int j = 0; j < M + 1; j++)
            z(i, j) = 0;
}

void solver::prepare(Matrix& v, type_d a, type_d c) {
  v.resize(N + 1, M + 1);
  buf.resize(N + 1, M + 1);

  for (int i = 0; i < N + 1; i++) {
    v(i, 0) = ux0(a + h * i);
    v(i, M) = ux1(a + h * i);
  }

  for (int i = 1; i < M; i++) {
    v(0, i) = u0y(c + k * i);
    v(M, i) = u1y(c + k * i);
  }

  for (int i = 1; i < M; i++) {
    buf(0, i) = v(0, i);
    buf(M, i) = v(M, i);
  }

  for (int i = 1; i < N; i++)
    for (int j = 1; j < M; j++)
      v(i, j) = ux0(a + h * i) + k * j * (ux1(a + h * i) - ux0(a + h * i));
}

void solver::step(Matrix& v, Matrix& z, type_d a, type_d c, type_d& mz, type_d& acc) {
    //zeidel

}

void solver::step(Matrix& v, type_d a, type_d c, type_d& acc) {
    //zeidel

}

void solver::step_mvr(Matrix& v, Matrix& z, type_d a, type_d c, type_d& mz, type_d& acc) {
    //mvr

}

void solver::step_mvr(Matrix& v, type_d a, type_d c, type_d& acc) {

}

void solver::step_msi(Matrix& v, Matrix& z, type_d a, type_d c, type_d& mz, type_d& acc, type_d& tau) {

  it++;
  int max_thr = omp_get_max_threads();
#pragma omp parallel for schedule(dynamic,N/max_thr+1)
  for (int i = 1; i < N; i++) {
    for (int j = 1; j < M; j++) {
      buf(i, j) = v(i, j) - tau * (A * v(i, j) + hor * v(i - 1, j)
        + hor * v(i + 1, j)
        + hor * v(i, j - 1)
        + hor * v(i, j + 1) - right_side(i, j));
    }
  }
  
  std::vector<double> accs(max_thr);
  std::vector<double> mzs(max_thr);
#pragma omp parallel for schedule(dynamic,N/max_thr+1)
  for (int i = 1; i < N; i++) {
    int nthr = omp_get_thread_num();
    double b;
    for (int j = 1; j < M; j++) {
      accs[nthr] = std::max(abs(buf(i, j) - v(i, j)), accs[nthr]);
      mzs[nthr] = std::max(abs(buf(i, j) - u(i * h, j * k)), mzs[nthr]);
      v(i, j) = buf(i, j);
    }
  }
  mz = mzs[0];
  acc = accs[0];
  for (int i = 1; i < max_thr; i++) {
    mz = std::max(mz, mzs[i]);
    acc = std::max(acc, accs[i]);
  }
}

void solver::step_msi(Matrix& v, type_d a, type_d c, type_d& acc, type_d& tau) {

  it++;
  int max_thr = omp_get_max_threads();
#pragma omp parallel for schedule(dynamic,N/max_thr+1)
  for (int i = 1; i < N; i++) {
    for (int j = 1; j < M; j++) {
      buf(i, j) = v(i, j) - tau * (A * v(i, j) + hor * v(i - 1, j)
        + hor * v(i + 1, j)
        + hor * v(i, j - 1)
        + hor * v(i, j + 1) - right_side(i, j));
    }
  }
  std::vector<double> accs(max_thr);
#pragma omp parallel for schedule(dynamic,N/max_thr+1)
  for (int i = 0; i < N + 1; i++) {
    int nthr = omp_get_thread_num();
    double b;
    for (int j = 0; j < M + 1; j++) {
      accs[nthr] = std::max(abs(buf(i, j) - v(i, j)), accs[nthr]);
      b = buf(i, j);
      buf(i, j) = v(i, j);
      v(i, j) = b;
    }
  }
  acc = accs[0];
  for (int i = 1; i < max_thr; i++) {
    acc = std::max(acc, accs[i]);
  }
}

void solver::copy(Matrix& v1, Matrix& z1, Matrix& v2, Matrix& z2) {
    for (int i = 0; i <= N; i++)
        for (int j = 0; j <= M; j++) {
            v2(i, j) = v1(i, j);
            z2(i, j) = z1(i, j);
        }
}

void solver::copy(Matrix& v1,Matrix& v2) {
    for (int i = 0; i <= N; i++)
        for (int j = 0; j <= M; j++) {
          v2(i, j) = v1(i, j);
        }
}

Q_INVOKABLE void solver::solve(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps, int m_it, std::vector<Matrix>& v, std::vector<Matrix>& z) {
    timer.start();
    N = n;
    M = m;
    x0 = a;
    X = b;
    y0 = c;
    Y = d;
    k = (Y - y0) / M;
    h = (X - x0) / N;
    valid = true;
    epsilon = eps;

    max_it = m_it;
    it = 0;
    hor = type_d(1) / (h * h);
    ver = type_d(1) / (k * k);
    A = type_d(- 2) * (type_d(1) / (h * h) + type_d(1) / (k * k));
    v.resize(10);
    z.resize(10);
    iter.resize(2 + max_it / interval);
    ACCURACY.resize(2 + max_it / interval);
    MAX_R.resize(2 + max_it / interval);
    MAX_Z.resize(2 + max_it / interval);

    int iter_size = 1;

    if (N < 100 && M < 100) {
        for(int i = 0; i < 10; i++) {
            prepare(v[i], z[i], a, c);
        }
    } else {
        prepare(v[9], z[9], a, c);
    }
    fill_right_side(v[9], a, c);
    type_d last_mz;
    type_d last_accuracy = type_d(0);
    int cur_photo = 1;

    emit progressUpdate(0, 66, timer.elapsed(), 1);
    meth = Methods::msi;
    if (meth == Methods::zeidel){
        timer.start();
        step(v[9], z[9], a, c, last_mz, last_accuracy);
        emit progressUpdate((0 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], z[9], v[cur_photo], z[cur_photo]);
        cur_photo++;

        iter[0] = (it);
        ACCURACY[0] = last_accuracy;
        calc_r(v[9]);
        MAX_R[0] = max_r;
        MAX_Z[0] = last_mz;

        step(v[9], z[9], a, c, last_mz, last_accuracy);
        emit progressUpdate((1 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], z[9], v[cur_photo], z[cur_photo]);
        cur_photo++;

        type_d cur_accuracy = last_accuracy;

        for (size_t i = 2; i < max_it && cur_accuracy > eps; i++) {
            step(v[9], z[9], a, c, last_mz, cur_accuracy);
            if(i % interval == 0){
                iter[iter_size] = (it);
                ACCURACY[iter_size] = (cur_accuracy);
                calc_r(v[9]);
                MAX_R[iter_size] = (max_r);
                MAX_Z[iter_size] = (last_mz);
                iter_size++;
            }
            if (cur_accuracy < (last_accuracy / type_d(2)) && cur_photo < 9 && N < 100 && M < 100) {
                copy(v[9], z[9], v[cur_photo], z[cur_photo]);
                last_accuracy = cur_accuracy;
                cur_photo++;
            }
            emit progressUpdate((i * 100) / max_it, cur_accuracy, timer.elapsed(), it);
        }
        emit progressUpdate(100, cur_accuracy, timer.elapsed(), it);

        if (N < 100 && M < 100) {
            for (; cur_photo < 9; cur_photo++) {
                copy(v[9], z[9], v[cur_photo], z[cur_photo]);
            }
        }
        max_z = last_mz;
        achieved_accuracy = cur_accuracy;
        calc_r(v[9]);
        iter[iter_size] = (it);
        ACCURACY[iter_size] = (achieved_accuracy);
        MAX_R[iter_size] = (max_r);
        MAX_Z[iter_size] = (max_z);
        iter.resize(iter_size);
        ACCURACY.resize(iter_size);
        MAX_R.resize(iter_size);
        MAX_Z.resize(iter_size);
    } else if (meth == Methods::mvr){
        timer.start();
        step_mvr(v[9], z[9], a, c, last_mz, last_accuracy);
        emit progressUpdate((0 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], z[9], v[cur_photo], z[cur_photo]);
        cur_photo++;

        iter[0] = (it);
        ACCURACY[0] = last_accuracy;
        calc_r(v[9]);
        MAX_R[0] = max_r;
        MAX_Z[0] = last_mz;

        step_mvr(v[9], z[9], a, c, last_mz, last_accuracy);
        emit progressUpdate((1 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], z[9], v[cur_photo], z[cur_photo]);
        cur_photo++;

        type_d cur_accuracy = last_accuracy;

        for (size_t i = 2; i < max_it && cur_accuracy > eps; i++) {
            step_mvr(v[9], z[9], a, c, last_mz, cur_accuracy);
            if(i % interval == 0){
                iter[iter_size] = (it);
                ACCURACY[iter_size] = (cur_accuracy);
                calc_r(v[9]);
                MAX_R[iter_size] = (max_r);
                MAX_Z[iter_size] = (last_mz);
                iter_size++;
            }
            if (cur_accuracy < (last_accuracy / type_d(2)) && cur_photo < 9 && N < 100 && M < 100) {
                copy(v[9], z[9], v[cur_photo], z[cur_photo]);
                last_accuracy = cur_accuracy;
                cur_photo++;
            }
            emit progressUpdate((i * 100) / max_it, cur_accuracy, timer.elapsed(), it);
        }
        emit progressUpdate(100, cur_accuracy, timer.elapsed(), it);

        if (N < 100 && M < 100) {
            for (; cur_photo < 9; cur_photo++) {
                copy(v[9], z[9], v[cur_photo], z[cur_photo]);
            }
        }
        max_z = last_mz;
        achieved_accuracy = cur_accuracy;
        calc_r(v[9]);
        iter[iter_size] = (it);
        ACCURACY[iter_size] = (achieved_accuracy);
        MAX_R[iter_size] = (max_r);
        MAX_Z[iter_size] = (max_z);
        iter.resize(iter_size);
        ACCURACY.resize(iter_size);
        MAX_R.resize(iter_size);
        MAX_Z.resize(iter_size);
    }
    else if (meth == Methods::msi) {
      type_d lambda1 = -4.0 / (h * h) * sin(boost::math::constants::pi<type_d>() / (2.0 * n)) - 4.0 / (k * k) * sin(boost::math::constants::pi<type_d>() / (2.0 * m));
      type_d lambdaN = -4.0 / (h * h) * sin(boost::math::constants::pi<type_d>() * (n - 1) / (2.0 * n)) - 4.0 / (k * k) * sin(boost::math::constants::pi<type_d>() * (m - 1) / (2.0 * m));
      lambda1 = A - abs(2.0 * hor + 2.0 * ver - A);
      lambdaN = A + abs(2.0 * hor + 2.0 * ver - A);
      lambda1 = A - abs(1.0 * hor + 2.0 * ver - A) < lambda1 ? A - abs(1.0 * hor + 2.0 * ver - A) : lambda1;
      lambdaN = A + abs(1.0 * hor + 2.0 * ver - A) > lambdaN ? A + abs(1.0 * hor + 2.0 * ver - A) : lambdaN;
      lambda1 = A - abs(2.0 * hor + 1.0 * ver - A) < lambda1 ? A - abs(2.0 * hor + 1.0 * ver - A) : lambda1;
      lambdaN = A + abs(2.0 * hor + 1.0 * ver - A) > lambdaN ? A + abs(2.0 * hor + 1.0 * ver - A) : lambdaN;
      lambda1 = A - abs(1.0 * hor + 1.0 * ver - A) < lambda1 ? A - abs(1.0 * hor + 1.0 * ver - A) : lambda1;
      lambdaN = A + abs(1.0 * hor + 1.0 * ver - A) > lambdaN ? A + abs(1.0 * hor + 1.0 * ver - A) : lambdaN;
      
      type_d tau(type_d(2.0) / (lambda1 + lambdaN));

      timer.start();
      step_msi(v[9], z[9], a, c, last_mz, last_accuracy, tau);
      emit progressUpdate((0 * 100) / max_it, last_accuracy, timer.elapsed(), it);
      if (N < 100 && M < 100) copy(v[9], z[9], v[cur_photo], z[cur_photo]);
      cur_photo++;

      iter[0] = (it);
      ACCURACY[0] = last_accuracy;
      calc_r(v[9]);
      MAX_R[0] = max_r;
      MAX_Z[0] = last_mz;

      step_msi(v[9], z[9], a, c, last_mz, last_accuracy, tau);
      emit progressUpdate((1 * 100) / max_it, last_accuracy, timer.elapsed(), it);
      if (N < 100 && M < 100) copy(v[9], z[9], v[cur_photo], z[cur_photo]);
      cur_photo++;

      type_d cur_accuracy = last_accuracy;

      for (size_t i = 2; i < max_it && cur_accuracy > eps; i++) {
        step_msi(v[9], z[9], a, c, last_mz, cur_accuracy, tau);
        if (i % interval == 0) {
          iter[iter_size] = (it);
          ACCURACY[iter_size] = (cur_accuracy);
          calc_r(v[9]);
          MAX_R[iter_size] = (max_r);
          MAX_Z[iter_size] = (last_mz);
          iter_size++;
        }
        if (cur_accuracy < (last_accuracy / type_d(2)) && cur_photo < 9 && N < 100 && M < 100) {
          copy(v[9], z[9], v[cur_photo], z[cur_photo]);
          last_accuracy = cur_accuracy;
          cur_photo++;
        }
        emit progressUpdate((i * 100) / max_it, cur_accuracy, timer.elapsed(), it);
      }
      emit progressUpdate(100, cur_accuracy, timer.elapsed(), it);

      if (N < 100 && M < 100) {
        for (; cur_photo < 9; cur_photo++) {
          copy(v[9], z[9], v[cur_photo], z[cur_photo]);
        }
      }
      max_z = last_mz;
      achieved_accuracy = cur_accuracy;
      calc_r(v[9]);
      iter[iter_size] = (it);
      ACCURACY[iter_size] = (achieved_accuracy);
      MAX_R[iter_size] = (max_r);
      MAX_Z[iter_size] = (max_z);
      iter.resize(iter_size);
      ACCURACY.resize(iter_size);
      MAX_R.resize(iter_size);
      MAX_Z.resize(iter_size);
    }

    duration = timer.elapsed();
    emit solveFinished();
}

Q_INVOKABLE void solver::solve(int n, int m, type_d a, type_d b, type_d c, type_d d, type_d eps, int m_it, std::vector<Matrix>& v) {
    timer.start();
    N = n;
    M = m;
    x0 = a;
    X = b;
    y0 = c;
    Y = d;
    k = (Y - y0) / M;
    h = (X - x0) / N;
    epsilon = eps;
    max_it = m_it;
    it = 0;
    hor = type_d(1) / (h * h);
    ver = type_d(1) / (k * k);
    A = type_d(- 2) * (type_d(1) / (h * h) + type_d(1) / (k * k));
    v.resize(10);
    iter.resize(2 + max_it / interval);
    ACCURACY.resize(2 + max_it / interval);
    MAX_R.resize(2 + max_it / interval);
    int iter_size = 1;

    if (N < 100 && M < 100) {
        for(int i = 0; i < 10; i++){
            prepare(v[i], a, c);
        }
    } else {
        prepare(v[9], a, c);
    }
    fill_right_side(v[9], a, c);
    type_d last_accuracy = type_d(0);
    int cur_photo = 1;

    emit progressUpdate(0, 66, timer.elapsed(), 1);
    meth = Methods::msi;
    if (meth == Methods::zeidel){
        timer.start();
        step(v[9], a, c, last_accuracy);
        emit progressUpdate((0 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], v[cur_photo]);
        cur_photo++;

        iter[0] = (it);
        ACCURACY[0] = last_accuracy;
        calc_r(v[9]);
        MAX_R[0] = max_r;

        step(v[9], a, c, last_accuracy);
        emit progressUpdate((1 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], v[cur_photo]);
        cur_photo++;

        type_d cur_accuracy = last_accuracy;

        for (size_t i = 2; i < max_it && cur_accuracy > eps; i++) {
            step(v[9], a, c, cur_accuracy);
            if(i % interval == 0){
                iter[iter_size] = (it);
                ACCURACY[iter_size] = (cur_accuracy);
                calc_r(v[9]);
                MAX_R[iter_size] = (max_r);
                iter_size++;
            }
            if (cur_accuracy < (last_accuracy / type_d(2)) && cur_photo < 9 && N < 100 && M < 100) {
                copy(v[9], v[cur_photo]);
                last_accuracy = cur_accuracy;
                cur_photo++;
            }
            emit progressUpdate((i * 100) / max_it, cur_accuracy, timer.elapsed(), it);
        }
        emit progressUpdate(100, cur_accuracy, timer.elapsed(), it);

        if (N < 100 && M < 100) {
            for (; cur_photo < 9; cur_photo++) {
                copy(v[9], v[cur_photo]);
            }
        }
        achieved_accuracy = cur_accuracy;
        calc_r(v[9]);
        iter[iter_size] = (it);
        ACCURACY[iter_size] = (cur_accuracy);
        MAX_R[iter_size] = (max_r);
        iter.resize(iter_size);
        ACCURACY.resize(iter_size);
        MAX_R.resize(iter_size);
    } else if(meth == Methods::mvr){
        timer.start();
        step_mvr(v[9], a, c, last_accuracy);
        emit progressUpdate((0 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], v[cur_photo]);
        cur_photo++;

        iter[0] = (it);
        ACCURACY[0] = last_accuracy;
        calc_r(v[9]);
        MAX_R[0] = max_r;

        step_mvr(v[9], a, c, last_accuracy);
        emit progressUpdate((1 * 100) / max_it, last_accuracy, timer.elapsed(), it);
        if (N < 100 && M < 100) copy(v[9], v[cur_photo]);
        cur_photo++;

        type_d cur_accuracy = last_accuracy;

        for (size_t i = 2; i < max_it && cur_accuracy > eps; i++) {
            step_mvr(v[9], a, c, cur_accuracy);
            if(i % interval == 0){
                iter[iter_size] = (it);
                ACCURACY[iter_size] = (cur_accuracy);
                calc_r(v[9]);
                MAX_R[iter_size] = (max_r);
                iter_size++;
            }
            if (cur_accuracy < (last_accuracy / type_d(2)) && cur_photo < 9 && N < 100 && M < 100) {
                copy(v[9], v[cur_photo]);
                last_accuracy = cur_accuracy;
                cur_photo++;
            }
            emit progressUpdate((i * 100) / max_it, cur_accuracy, timer.elapsed(), it);
        }
        emit progressUpdate(100, cur_accuracy, timer.elapsed(), it);

        if (N < 100 && M < 100) {
            for (; cur_photo < 9; cur_photo++) {
                copy(v[9], v[cur_photo]);
            }
        }
        achieved_accuracy = cur_accuracy;
        calc_r(v[9]);
        iter[iter_size] = (it);
        ACCURACY[iter_size] = (cur_accuracy);
        MAX_R[iter_size] = (max_r);
        iter.resize(iter_size);
        ACCURACY.resize(iter_size);
        MAX_R.resize(iter_size);
    }
    else if (meth == Methods::msi) {

    type_d lambda1 = 4.0 / (h * h) * sin(boost::math::constants::pi<type_d>() / (2.0 * n)) + 4.0 / (k * k) * sin(boost::math::constants::pi<type_d>() / (2.0 * m));
    type_d lambdaN = 4.0 / (h * h) * sin(boost::math::constants::pi<type_d>() * (n - 1) / (2.0 * n)) + 4.0 / (k * k) * sin(boost::math::constants::pi<type_d>() * (m - 1) / (2.0 * m));
    lambda1 = A - abs(2.0 * hor + 2.0 * ver - A);
    lambdaN = A + abs(2.0 * hor + 2.0 * ver - A);
    lambda1 = A - abs(1.0 * hor + 2.0 * ver - A) < lambda1 ? A - abs(1.0 * hor + 2.0 * ver - A) : lambda1;
    lambdaN = A + abs(1.0 * hor + 2.0 * ver - A) > lambdaN ? A + abs(1.0 * hor + 2.0 * ver - A) : lambdaN;
    lambda1 = A - abs(2.0 * hor + 1.0 * ver - A) < lambda1 ? A - abs(2.0 * hor + 1.0 * ver - A) : lambda1;
    lambdaN = A + abs(2.0 * hor + 1.0 * ver - A) > lambdaN ? A + abs(2.0 * hor + 1.0 * ver - A) : lambdaN;
    lambda1 = A - abs(1.0 * hor + 1.0 * ver - A) < lambda1 ? A - abs(1.0 * hor + 1.0 * ver - A) : lambda1;
    lambdaN = A + abs(1.0 * hor + 1.0 * ver - A) > lambdaN ? A + abs(1.0 * hor + 1.0 * ver - A) : lambdaN;

      type_d tau(-type_d(2.0) / (lambda1 + lambdaN));

      timer.start();
      step_msi(v[9], a, c, last_accuracy, tau);
      emit progressUpdate((0 * 100) / max_it, last_accuracy, timer.elapsed(), it);
      if (N < 100 && M < 100) copy(v[9], v[cur_photo]);
        cur_photo++;

      iter[0] = (it);
      ACCURACY[0] = last_accuracy;
      calc_r(v[9]);
      MAX_R[0] = max_r;

      step_msi(v[9], a, c, last_accuracy, tau);
      emit progressUpdate((1 * 100) / max_it, last_accuracy, timer.elapsed(), it);
      if (N < 100 && M < 100) copy(v[9], v[cur_photo]);
      cur_photo++;

      type_d cur_accuracy = last_accuracy;

      for (size_t i = 2; i < max_it && cur_accuracy > eps; i++) {
        step_msi(v[9], a, c, cur_accuracy, tau);
        if (i % interval == 0) {
          iter[iter_size] = (it);
          ACCURACY[iter_size] = (cur_accuracy);
          calc_r(v[9]);
          MAX_R[iter_size] = (max_r);
          iter_size++;
        }
        if (cur_accuracy < (last_accuracy / type_d(2)) && cur_photo < 9 && N < 100 && M < 100) {
          copy(v[9], v[cur_photo]);
          last_accuracy = cur_accuracy;
          cur_photo++;
        }
        emit progressUpdate((i * 100) / max_it, cur_accuracy, timer.elapsed(), it);
      }
      emit progressUpdate(100, cur_accuracy, timer.elapsed(), it);

      if (N < 100 && M < 100) {
        for (; cur_photo < 9; cur_photo++) {
          copy(v[9], v[cur_photo]);
        }
      }
      achieved_accuracy = cur_accuracy;
      calc_r(v[9]);
      iter[iter_size] = (it);
      ACCURACY[iter_size] = (cur_accuracy);
      MAX_R[iter_size] = (max_r);
      iter.resize(iter_size);
      ACCURACY.resize(iter_size);
      MAX_R.resize(iter_size);
    }
    duration = timer.elapsed();
    valid = true;
    emit solveFinished();
}

inline int solver::is_border(int i, int j){
    return ((i == 0) || (j == 0) || (i == N) || (j == M)) ? 1 : 0;
}

void solver::fill_right_side(Matrix& v, type_d a, type_d c){
  right_side = Matrix(N + 1, M + 1);
      for (int j = 1; j < M; j++) {
        for (int i = 1; i < N; i++) {
          right_side(i, j) = (-f(a + i * h, c + j * k));
        }
      }
}

void solver::calc_r_vec(Matrix& v, std::vector<type_d>& res) {
  int place = 0;
}

void solver::calc_r(Matrix& v){
    int place = 0;
    type_d r = type_d(0);
#pragma omp parallel for schedule(dynamic,100)
    
      for (int j = 1; j < M; j++) {
        for (int i = 1; i < N; i++) {
          buf(i, j) = abs((A * v(i, j) + (hor * v(i - 1, j)
            +  hor * v(i + 1, j)
            +  hor * v(i, j - 1)
            +  hor * v(i, j + 1) - right_side(i, j))));
        }
      }
      for (int j = 1; j < M; j++) {
        for (int i = 1; i < N; i++) {
          r = std::max(r, buf(i, j));
        }
      }
    
    max_r = r;
}
