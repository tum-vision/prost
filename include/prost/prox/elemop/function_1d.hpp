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

namespace prost {

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

template<typename T>
struct Function1DLq
{
  // |x|^alpha for any alpha > 0.
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    return 0.; // TODO: Implementation
  }
};

template<typename T>
struct Function1DTruncQuad
{
  // Truncated quadratic function, min(|x|^2, alpha)
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    return 0; // TODO: Implementation
  }
};

template<typename T>
struct Function1DTruncLinear
{
  // Truncated linear function, min(|x|, alpha)
  inline __host__ __device__
  T
  operator()(T x0, T tau, T alpha, T beta) const
  {
    return 0; // TODO: Implementation
  }
};

// TODO: Implement the following prox-Functions
// identity,-log(x),xlog(x),log(1+exp(x)),1/x,exp(x),max(0,-x)
// max(0,x)

} // namespace prost

#endif // PROST_FUNCTION_1D_HPP_

