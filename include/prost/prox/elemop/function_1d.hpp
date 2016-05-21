/**
* This file is part of prost.
*
* Copyright 2016 Thomas Möllenhoff <thomas dot moellenhoff at in dot tum dot de> 
* and Emanuel Laude <emanuel dot laude at in dot tum dot de> (Technical University of Munich)
*
* prost is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* prost is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with prost. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PROST_FUNCTION_1D_HPP_
#define PROST_FUNCTION_1D_HPP_

#include <limits>

// TODO: Implement the following prox-Functions
// identity,-log(x),xlog(x),log(1+exp(x)),1/x,exp(x),max(0,-x), 1/x

namespace prost {

using std::numeric_limits;

// implementation of 1D prox operators
template<typename T>
struct Function1DZero
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    return x0;
  }
};

template<typename T> 
struct Function1DAbs
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    if(x0 >= tau)
      return x0 - tau;
    else if(x0 <= -tau)
      return x0 + tau;

    return 0;
  }
};

template<typename T>
struct Function1DSquare
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    return x0 / (1. + tau);
  }
};

template<typename T>
struct Function1DIndLeq0
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    if(x0 > 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Function1DIndGeq0
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Function1DIndEq0
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    return 0.;
  }
};

template<typename T>
struct Function1DIndBox01
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    if(x0 > 1.)
      return 1.;
    else if(x0 < 0.)
      return 0.;

    return x0;
  }
};

template<typename T>
struct Function1DMaxPos0
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    if(x0 > tau)
      return x0 - tau;
    else if(x0 < 0.)
      return x0;

    return 0.;
  }
};

template<typename T>
struct Function1DL0
{
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    if(x0*x0 > 2 * tau)
      return x0;

    return 0;
  }
};

template<typename T>
struct Function1DHuber
{
  // min_x huber_alpha(x) + (1/2tau) (x-x0)^2
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    T result = (x0 / tau) / (1. + alpha / tau);
    result /= max(static_cast<T>(1), abs(result));  
    return x0 - tau * result;
  }
};

/// \brief Solve 0.5 * (t-1)^2 + \alpha t^q using Newton's method
template<typename T>
inline __host__ __device__ 
T solveLqGeneralNewton(const T t0, const T alpha, const T q, const T eps) 
{
  T t = t0;
  T delta = 0;
  do
  {
    // calculate first and second derivatives
    const T power = pow(t, q);
    const T dF1 = t - 1 + alpha * q * power / t;
    const T dF2 = 1 + alpha * q * (q - 1) * power / (t * t);

    // newton step
    delta = dF1 / dF2;
    t = t - delta;
  } while(delta > eps); 

  return t;
}

/// \brief Solve 0.5 * (t-1)^2 + \alpha t^q analytically for q=0.5
template<typename T>
inline __host__ __device__ 
T solveLqHalfAnalytic(const T alpha) 
{
  const T sqrt3 = sqrt(static_cast<T>(3));
  const T PI_half = static_cast<T>(1.5707963267948966192313216916397514420985846996875529);
  const T s = 2 * (sin(static_cast<T>((acos(static_cast<T>(alpha * 3 * sqrt3 / 4)) + PI_half) / 3))) / sqrt3;
  return s * s;
}

template<typename T>
struct Function1DLq
{
  T eps;

  __host__ __device__ Function1DLq();

  // |x|^alpha for any alpha >= 0.
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    if(alpha == 1)
    {
      Function1DAbs<T> fn_abs;
      return fn_abs(x0, tau, alpha, beta);
    }
    else if(alpha == 0)
    {
      Function1DL0<T> fn_l0;
      return fn_l0(x0, tau, alpha, beta);
    }
    else // general case
    {
      T t = 0;

      if(abs(x0) > 0) 
      {
        T factor = tau * pow(abs(x0), static_cast<T>(alpha - 2));

        if(alpha < 1) 
        {
          // in nonconvex case we have to check if value at 
          // boundary is better or problem doesn't even have stationary point
          const T t2 = 2 * (alpha - 1) / (alpha - 2);
          if(factor < 0.5 * (1 - (t2 - 1) * (t2 - 1)) / pow(t2, alpha))
          {
            if(alpha == 0.5)
              t = solveLqHalfAnalytic<T>(factor);
            else
              t = solveLqGeneralNewton<T>(1, factor, alpha, eps); 
          }
        }
        else
        {
          t = solveLqGeneralNewton<T>(1, factor, alpha, eps);
        }
      }

      return t * abs(x0);
    }
  }
};

template<>
inline __host__ __device__ 
Function1DLq<float>::Function1DLq()
{
  eps = 1e-5;
}

template<>
inline __host__ __device__ 
Function1DLq<double>::Function1DLq()
{
  eps = 1e-11;
}

template<typename T>
struct Function1DTruncQuad
{
  // Truncated quadratic function, min(alpha |x|^2, beta)
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    Function1DSquare<T> fn_sq;

    const T x_sq = fn_sq(x0, 2 * tau * alpha, 0, 0);
    const T en_sq = alpha * x_sq * x_sq + (x_sq - x0) * (x_sq - x0) / (2 * tau);

    if(en_sq < beta)
      return x_sq;
    else
      return x0;
  }
};

template<typename T>
struct Function1DTruncLinear
{
  // Truncated linear function, min(alpha |x|, beta)
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    Function1DAbs<T> fn_abs;

    const T x_shrink = fn_abs(x0, tau * alpha, 0, 0);
    const T en_shrink = (x_shrink - x0) * (x_shrink - x0) / (2 * tau) + alpha * abs(x_shrink);

    if(en_shrink < beta)
      return x_shrink;
    else
      return x0;
  }
};

} // namespace prost

#endif // PROST_FUNCTION_1D_HPP_

