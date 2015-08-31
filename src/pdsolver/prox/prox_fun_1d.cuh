/*
 * This file is part of pdsolver.
 *
 * Copyright (C) 2015 Thomas Möllenhoff <thomas.moellenhoff@in.tum.de> 
 *
 * pdsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pdsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pdsolver. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef PROX_FUN_1D_CUH
#define PROX_FUN_1D_CUH

template<typename real>
struct prox_1d_zero
{
  inline __host__ __device__ real apply(real x0, real tau) const
  {
    return x0;
  }
};

template<typename real> 
struct prox_1d_abs
{
  inline __host__ __device__ real apply(real x0, real tau) const
  {
    if(x0 >= real(1))
      return x0 - real(1);
    else if(x0 <= real(-1))
      return x0 + real(1);
    else
      return real(0);
  }
};

template<typename real>
struct prox_1d_square
{
  inline __host__ __device__ real apply(real x0, real tau) const
  {
    return x0 / (real(1) + tau);
  }
};

template<typename real>
struct prox_1d_leq
{
  inline __host__ __device__ real apply(real x0, real tau) const
  {
    if(x0 > real(1))
      return real(1);

    return x0;
  }
};

template<typename real>
struct prox_1d_eq
{
  inline __host__ __device__ real apply(real x0, real tau) const
  {
    return real(0);
  }
};

template<typename real>
struct prox_1d_absleq
{
  inline __host__ __device__ real apply(real x0, real tau) const
  {
    if(x0 > real(1))
      return real(1);
    else if(x0 < real(-1))
      return real(-1);

    return x0;
  }
};


#endif
