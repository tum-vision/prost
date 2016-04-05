////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2015, Christopher Fougner                                    //
// All rights reserved.                                                       //
//                                                                            //
// Redistribution and use in source and binary forms, with or without         //
// modification, are permitted provided that the following conditions are     //
// met:                                                                       //
//                                                                            //
//   1. Redistributions of source code must retain the above copyright        //
//      notice, this list of conditions and the following disclaimer.         //
//                                                                            //
//   2. Redistributions in binary form must reproduce the above copyright     //
//      notice, this list of conditions and the following disclaimer in the   //
//      documentation and/or other materials provided with the distribution.  //
//                                                                            //
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS        //
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED  //
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR //
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR          //
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,      //
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,        //
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR         //
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF     //
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING       //
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS         //
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.               //
////////////////////////////////////////////////////////////////////////////////

//  CGLS Conjugate Gradient Least Squares
//  Attempts to solve the least squares problem
//
//    min. ||Ax - b||_2^2 + s ||x||_2^2
//
//  using the Conjugate Gradient for Least Squares method. This is more stable
//  than applying CG to the normal equations. Supports both generic operators
//  for computing Ax and A^Tx as well as a sparse matrix version.
//
//  ------------------------------ GENERIC  ------------------------------------
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  F          - Generic GEMV-like functor type with signature
//               int gemv(char op, T alpha, const T *x, T beta, T *y). Upon
//               exit, y should take on the value y := alpha*op(A)x + beta*y.
//               If successful the functor must return 0, otherwise a non-zero
//               value should be returned.
//
//  Function Arguments:
//  A          - Operator that computes Ax and A^Tx.
//
//  (m, n)     - Matrix dimensions of A.
//
//  b          - Pointer to right-hand-side vector.
//
//  x          - Pointer to solution. This vector will also be used as an
//               initial guess, so it must be initialized (eg. to 0).
//
//  shift      - Regularization parameter s. Solves (A'*A + shift*I)*x = A'*b.
//
//  tol        - Specifies tolerance (recommended 1e-6).
//
//  maxit      - Maximum number of iterations (recommended > 100).
//
//  quiet      - Disable printing to console.
//
//  ------------------------------ SPARSE --------------------------------------
//
//  Template Arguments:
//  T          - Data type (float or double).
//
//  O          - Sparse ordering (cgls::CSC or cgls::CSR).
//
//  Function Arguments:
//  val        - Array of matrix values. The array should be of length nnz.
//
//  ptr        - Column pointer if (O is CSC) or row pointer if (O is CSR).
//               The array should be of length m+1.
//
//  ind        - Row indices if (O is CSC) or column indices if (O is CSR).
//               The array should be of length nnz.
//
//  (m, n)     - Matrix dimensions of A.
//
//  nnz        - Number of non-zeros in A.
//
//  b          - Pointer to right-hand-side vector.
//
//  x          - Pointer to solution. This vector will also be used as an
//               initial guess, so it must be initialized (eg. to 0).
//
//  shift      - Regularization parameter s. Solves (A'*A + shift*I)*x = A'*b.
//
//  tol        - Specifies tolerance (recommended 1e-6).
//
//  maxit      - Maximum number of iterations (recommended > 100).
//
//  quiet      - Disable printing to console.
//
//  ----------------------------------------------------------------------------
//
//  Returns:
//  0 : CGLS converged to the desired tolerance tol within maxit iterations.
//  1 : The vector b had norm less than eps, solution likely x = 0.
//  2 : CGLS iterated maxit times but did not converge.
//  3 : Matrix (A'*A + shift*I) seems to be singular or indefinite.
//  4 : Likely instable, (A'*A + shift*I) indefinite and norm(x) decreased.
//  5 : Error in applying operator A.
//  6 : Error in applying operator A^T.
//
//  Reference:
//  http://web.stanford.edu/group/SOL/software/cgls/
//

#ifndef CGLS_HPP_
#define CGLS_HPP_

#include <assert.h>
#include <stdio.h>

#include <cublas_v2.h>
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/transform_reduce.h>

#include <algorithm>
#include <limits>

namespace cgls {

// Data type for indices. Don't change this unless Nvidia some day
// changes their API (a la MKL).
typedef int INT;

// File-level functions and classes.
namespace {

// Axpy function.
inline cublasStatus_t axpy(cublasHandle_t handle, INT n, double *alpha,
  const double *x, INT incx, double *y, INT incy) {
  cublasStatus_t err = cublasDaxpy(handle, n, alpha, x, incx, y, incy);

  return err;
}

inline cublasStatus_t axpy(cublasHandle_t handle, INT n, float *alpha,
  const float *x, INT incx, float *y, INT incy) {
  cublasStatus_t err = cublasSaxpy(handle, n, alpha, x, incx, y, incy);

  return err;
}

inline cublasStatus_t axpy(cublasHandle_t handle, INT n, cuDoubleComplex *alpha,
  const cuDoubleComplex *x, INT incx,
  cuDoubleComplex *y, INT incy) {
  cublasStatus_t err = cublasZaxpy(handle, n, alpha, x, incx, y, incy);

  return err;
}

inline cublasStatus_t axpy(cublasHandle_t handle, INT n, cuFloatComplex *alpha,
  const cuFloatComplex *x, INT incx, cuFloatComplex *y,
  INT incy) {
  cublasStatus_t err = cublasCaxpy(handle, n, alpha, x, incx, y, incy);

  return err;
}

// 2-Norm based on thrust, potentially not as stable as cuBLAS version.
template <typename T>
struct NormSquared : thrust::unary_function<T, double> {
  inline __device__ double operator()(const T &x);
};
template <>
inline __device__ double NormSquared<double>::operator()(const double &x) {
  return x * x;
}

template <>
inline __device__ double NormSquared<float>::operator()(const float &x) {
  return static_cast<double>(x) * static_cast<double>(x);
}

template <typename T>
void nrm2(cublasHandle_t hdl, INT n, const T *x, double *result) {
  *result = sqrt(thrust::transform_reduce(thrust::device_pointer_cast(x),
      thrust::device_pointer_cast(x + n), NormSquared<T>(), 0.,
      thrust::plus<double>()));
}

// Casting from double to float, double.
template <typename T>
T StaticCast(double x);

template <>
inline double StaticCast<double>(double x) {
  return x;
}

template <>
inline float StaticCast<float>(double x) {
  return static_cast<float>(x);
}

// Numeric limit epsilon for float, double, complex_float, and complex_double.
template <typename T>
double Epsilon();

template<>
inline double Epsilon<double>() {
  return std::numeric_limits<double>::epsilon();
}

template<>
inline double Epsilon<float>() {
  return std::numeric_limits<float>::epsilon();
}

}  // namespace

// Conjugate Gradient Least Squares.
template <typename T, typename F>
int Solve(
  cublasHandle_t handle, 
  const F& A, 
  const INT m, 
  const INT n,
  const thrust::device_vector<T>& b, 
  thrust::device_vector<T>& x,
  const double shift, 
  const double tol,
  const int maxit, 
  bool quiet,
  thrust::device_vector<T>& p, // size=n
  thrust::device_vector<T>& q, // size=m
  thrust::device_vector<T>& r, // size=m
  thrust::device_vector<T>& s) // size=n
{
  // Variable declarations.
  double gamma, normp, normq, norms, norms0, normx, xmax;
  char fmt[] = "%5d %9.2e %12.5g\n";
  int err = 0, k = 0, flag = 0, indefinite = 0;

  // Constant declarations.
  const T kNegOne   = StaticCast<T>(-1.);
  const T kZero     = StaticCast<T>( 0.);
  const T kOne      = StaticCast<T>( 1.);
  const T kNegShift = StaticCast<T>(-shift);
  const double kEps = Epsilon<T>();

  thrust::copy(b.begin(), b.end(), r.begin());
  thrust::copy(x.begin(), x.end(), s.begin());

  // r = b - A*x.
  nrm2(handle, n, thrust::raw_pointer_cast(x.data()), &normx);
  cudaDeviceSynchronize();

  if (normx > 0.) {
    err = A('n', kNegOne, x, kOne, r);
    cudaDeviceSynchronize();

    if (err)
      flag = 5;
  }

  // s = A'*r - shift*x.
  err = A('t', kOne, r, kNegShift, s);
  cudaDeviceSynchronize();

  if (err)
    flag = 6;

  // Initialize.
  thrust::copy(s.begin(), s.end(), p.begin());

  nrm2(handle, n, thrust::raw_pointer_cast(s.data()), &norms);
  norms0 = norms;
  gamma = norms0 * norms0;
  nrm2(handle, n, thrust::raw_pointer_cast(x.data()), &normx);
  xmax = normx;
  cudaDeviceSynchronize();

  if (norms < kEps)
    flag = 1;

  if (!quiet)
    printf("    k     normx        resNE\n");

  for (k = 0; k < maxit && !flag; ++k) {
    // q = A * p.
    err = A('n', kOne, p, kZero, q);
    cudaDeviceSynchronize();

    if (err) {
      flag = 5;
      break;
    }

    // delta = norm(p)^2 + shift*norm(q)^2.
    nrm2(handle, n, thrust::raw_pointer_cast(p.data()), &normp);
    nrm2(handle, m, thrust::raw_pointer_cast(q.data()), &normq);
    cudaDeviceSynchronize();

    double delta = normq * normq + shift * normp * normp;

    if (delta <= 0.)
      indefinite = 1;
    if (delta == 0.)
      delta = kEps;
    T alpha = StaticCast<T>(gamma / delta);
    T neg_alpha = StaticCast<T>(-gamma / delta);

    // x = x + alpha*p.
    // r = r - alpha*q.
    axpy(handle, n, &alpha, thrust::raw_pointer_cast(p.data()), 1, 
      thrust::raw_pointer_cast(x.data()), 1);
    axpy(handle, m, &neg_alpha, thrust::raw_pointer_cast(q.data()), 1, 
      thrust::raw_pointer_cast(r.data()), 1);
    cudaDeviceSynchronize();

    // s = A'*r - shift*x.
    thrust::copy(x.begin(), x.end(), s.begin());

    err = A('t', kOne, r, kNegShift, s);
    cudaDeviceSynchronize();

    if (err) {
      flag = 6;
      break;
    }

    // Compute beta.
    nrm2(handle, n, thrust::raw_pointer_cast(s.data()), &norms);
    cudaDeviceSynchronize();

    double gamma1 = gamma;
    gamma = norms * norms;
    T beta = StaticCast<T>(gamma / gamma1);

    // p = s + beta*p.
    axpy(handle, n, &beta, thrust::raw_pointer_cast(p.data()), 1, 
      thrust::raw_pointer_cast(s.data()), 1);
    thrust::copy(s.begin(), s.end(), p.begin());
    cudaDeviceSynchronize();

    // Convergence check.
    nrm2(handle, n, thrust::raw_pointer_cast(x.data()), &normx);
    cudaDeviceSynchronize();

    xmax = std::max(xmax, normx);
    bool converged = (norms <= norms0 * tol) || (normx * tol >= 1.);
    if (!quiet && (converged || k % 10 == 0))
      printf(fmt, k, normx, norms / norms0);
    if (converged)
      break;
  }

  // Determine exit status.
  double shrink = normx / xmax;
  if (k == maxit)
    flag = 2;
  else if (indefinite)
    flag = 3;
  else if (shrink * shrink <= tol)
    flag = 4;

  //printf("CG took %d iterations.\n", k); 

  return flag;
}

}  // namespace cgls

#endif  // CGLS_HPP_

