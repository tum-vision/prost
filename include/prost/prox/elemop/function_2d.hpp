#ifndef PROST_FUNCTION_2D_HPP_
#define PROST_FUNCTION_2D_HPP_

// TODO: rename to Prox2D? not really a function.

namespace prost {

// implementation of 1D prox operators
template<typename T, class FUN_1D>
struct Function2DSum1D
{
  inline __host__ __device__
  void
  operator()(T y1, T y2, T& x1, T& x2, T tau, T alpha, T beta) const
  {
    FUN_1D fun_1d;
    x1 = fun_1d(y1, tau, alpha, beta);
    x2 = fun_1d(y2, tau, alpha, beta);
  }
};

template<typename T>
struct Function2DIndL1Ball
{
  inline __host__ __device__
  void
  operator()(T y1, T y2, T& x1, T& x2, T tau, T alpha, T beta) const
  {
    T v1 = abs(y1);
    T v2 = abs(y2);

    if(v1 + v2 <= alpha) {
      x1 = y1;
      x2 = y2;
      return;
    }

    // Project (v1, v2) onto the unit 2d-unit-simplex
    T mu1, mu2;
    if(v1 < v2) {
      mu1 = v2;
      mu2 = v1;
    } else {
      mu1 = v1;
      mu2 = v2;
    }

    T l = 0.5 * (mu2 - mu1 + alpha);
    char rho = 2;
    if(l <= 0.)
      rho = 1;

    T theta = (1./rho) * (mu1 + (rho == 2 ? mu2 : 0.) - alpha);


    mu1 = max(v1 - theta, 0.);
    mu2 = max(v2 - theta, 0.);

    // recover sign
    x1 = ((T(0) < y1) - (y1 < T(0))) * mu1;
    x2 = ((T(0) < y2) - (y2 < T(0))) * mu2;
  }
};


template<typename T, class OTHER_FUN_2D>
struct Function2DMoreau
{
  inline __host__ __device__
  void
  operator()(T y1, T y2, T& x1, T& x2, T tau, T alpha, T beta) const
  {
    OTHER_FUN_2D other_fun;
    T r1;
    T r2;
    other_fun(y1/tau, y2/tau, r1, r2, 1/tau, alpha, beta);

    x1 = y1 - tau * r1;
    x2 = y2 - tau * r2;
  }
};


} // namespace prost

#endif // PROST_FUNCTION_2D_HPP_

