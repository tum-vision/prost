#ifndef CUWRAP_HPP_
#define CUWRAP_HPP_

#include <cublas_v2.h>

namespace cuwrap
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
template<typename T> inline __host__ __device__ T max(const T& x, const T& y);

template<> inline __host__ __device__ float sqrt(const float& x) { return sqrtf(x); }
template<> inline __host__ __device__ double sqrt(const double& x) { return sqrt(x); }
template<> inline __host__ __device__ float max(const float& x, const float& y) { return fmaxf(x, y); }
template<> inline __host__ __device__ double max(const double& x, const double& y) { return fmax(x, y); }

}

#endif
