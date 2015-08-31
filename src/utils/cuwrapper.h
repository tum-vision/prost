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

#ifndef CUWRAPPER_H
#define CUWRAPPER_H

#include <cublas_v2.h>

namespace cuwrapper
{

// functions
template<typename T>
inline T nrm2(cublasHandle_t handle,T *x, int n);

template<typename T>
inline void scal(cublasHandle_t handle, T *x, const T factor, int n);

template<typename T>
inline void asum(cublasHandle_t handle, T *x, int n, T *result);

// specializations
template<>
inline float nrm2(cublasHandle_t handle, float *x, int n) {
  float result;
  cublasSnrm2(handle, n, x, 1, &result);

  return result;
}

template<>
inline double nrm2(cublasHandle_t handle, double *x, int n) {
  double result;
  cublasDnrm2(handle, n, x, 1, &result);

  return result;
}

template<>
inline void scal(cublasHandle_t handle, float *x, const float fac, int n) {
  cublasSscal(handle, n, &fac, x, 1);
}

template<>
inline void scal(cublasHandle_t handle, double *x, const double fac, int n) {
  cublasDscal(handle, n, &fac, x, 1);
}

template<>
inline void asum(cublasHandle_t handle, float *x, int n, float *result) {
  cublasSasum(handle, n, x, 1, result);
}

template<>
inline void asum(cublasHandle_t handle, double *x, int n, double *result) {
  cublasDasum(handle, n, x, 1, result);
}


// math functions
template<typename T> inline __host__ __device__ T sqrt(const T& x);

template<> inline __host__ __device__ float sqrt(const float& x) { return sqrtf(x); }
template<> inline __host__ __device__ double sqrt(const double& x) { return sqrt(x); }

}

#endif
