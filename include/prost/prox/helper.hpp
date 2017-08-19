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

#ifndef PROST_HELPER_HPP_
#define PROST_HELPER_HPP_

#include "prost/prox/vector.hpp"

namespace prost {

/// \brief Namspace grouping prox helper functions.
namespace helper {

template<typename T>
inline __host__ __device__ void swap(T& a, T& b)
{
  T c = a;
  a = b;
  b = c;
}

///
/// \brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
///        the parabola y >= \alpha ||x||^2 with \alpha > 0.
///
/// 
template<typename T>
inline __host__ __device__ void  ProjectEpiQuadNd(
     const Vector<T>& x0, const T y0, const T alpha, Vector<T>& x, T& y, size_t dim)
{

  T sq_norm_x0 = static_cast<T>(0);
  for(size_t i = 0; i < dim; i++) {
    sq_norm_x0 += x0[i] * x0[i];
  }
  const T norm_x0 = sqrt(sq_norm_x0);
  
  // nothing to do?
  if(y0 >= alpha * sq_norm_x0) {
    for(size_t i = 0; i < dim; i++) 
      x[i] = x0[i];

    y = y0;
  }
  else {
    const T a = 2. * alpha * norm_x0;
    const T b = 2. * (1. - 2. * alpha * y0) / 3.;
    T d, v;

    if(b < 0) {
      const T sq = pow(-b, static_cast<T>(3. / 2.));
      d = (a - sq) * (a + sq);      
    }
    else {
      d = a * a + b * b * b;
    }

    if(d >= 0) {
      const T c = pow(a + sqrt(d), static_cast<T>(1. / 3.));

      if(abs(c) > 1e-6)
        v = c - b / c;
      else
        v = 0;
    }
    else {
      v = 2 * sqrt(-b) * cos(acos(a / pow(-b, static_cast<T>(3. / 2.))) / static_cast<T>(3.));
    }

    if(norm_x0 > 0) { 
      for(size_t i = 0; i < dim; i++) {
        x[i] = (v / (2. * alpha)) * (x0[i] / norm_x0);
      }
    }
    else {
      for(size_t i = 0; i < dim; i++) {
        x[i] = 0;
      }
    }

    T sq_norm_x = static_cast<T>(0);
    for(size_t i = 0; i < dim; i++) {
      sq_norm_x += x[i] * x[i];
    }

    y = alpha * sq_norm_x;
   
  }
}


///
/// \brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
///        the parabola y >= \alpha x^2.
/// 
template<typename T>
inline __host__ __device__ void ProjectEpiQuad1d(
  T x0,
  T y0,
  T alpha,
  T& x,
  T& y)
{
  // nothing to do?
  if(y0 >= alpha * (x0 * x0)) {
    x = x0;
    y = y0;
  }
  else {
    const T a = 2. * alpha * abs(x0);
    const T b = 2. * (1. - 2. * alpha * y0) / 3.;
    T d, v;
    
    if(b < 0) {
      const T sq = pow(-b, static_cast<T>(3. / 2.));
      d = (a - sq) * (a + sq);      
    }
    else {
      d = a * a + b * b * b;
    }
    
    if(d >= 0) {
      const T c = pow(a + sqrt(d), static_cast<T>(1. / 3.));
      v = c - b / c;
    }
    else {
      v = 2 * sqrt(-b) * cos(acos(a / pow(-b, static_cast<T>(3. / 2.))) / static_cast<T>(3.));
    }

    if(x0 > 0) 
      x = v / (2. * alpha);
    else if(x0 < 0) 
      x = -v / (2. * alpha);
    else 
      x = 0;

    y = alpha * x * x;
  }
}

/// 
/// \brief Computes orthogonal projection of (x0, y0) onto the epigraph of the
///        parabola y >= p * x^2 + q * x + r.
/// 
template<typename T>
inline __host__ __device__ void ProjectEpiQuadGeneral1d(
  const T& x0,
  const T& y0,
  const T& p,
  const T& q,
  const T& r,
  T& x,
  T& y)
{
  T tildex;
  T tildey;
  
  helper::ProjectEpiQuad1d<T>(
    x0 + q / (2. * p),
    y0 + q * q / (4. * p) - r,
    p,
    tildex,
    tildey);

  x = tildex - q / (2. * p);
  y = tildey - q * q / (4. * p) + r;
}
  
/// 
/// \brief Computes projection of the d-dimensional vector v onto the
///        halfspace described by { x | <n, x> = t }.
/// 
template<typename T, typename ARRAY, typename CONST_ARRAY>
inline __host__ __device__ void ProjectHalfspace(
  CONST_ARRAY const& v,
  CONST_ARRAY const& n,
  T t,
  ARRAY& result, // TODO: this should be a const-reference, but doesn't compile. Why?
  int dim)
{
  T dot = 0, sq_norm = 0;

  for(int i = 0; i < dim; i++) {
    dot += n[i] * v[i];
    sq_norm += n[i] * n[i];
  }

  const T s = (dot - t) / sq_norm;
  
  for(int i = 0; i < dim; i++) 
    result[i] = v[i] - s * n[i];
}

/// 
/// \brief Checks whether a d-dimensional point v lies within the halfspace
///        described by a point p and normal n.
/// 
template<typename T, typename CONST_ARRAY>
inline __host__ __device__ bool IsPointInHalfspace(
  CONST_ARRAY const& v,
  CONST_ARRAY const& p,
  CONST_ARRAY const& n,
  int dim)
{
  T dot = 0;
  for(int i = 0; i < dim; i++) 
    dot += n[i] * (v[i] - p[i]);
  
  return dot <= 0.;
}

// partial function template specialization not allowed by C++ standard
// => use overloading instead.
template<typename T>
inline __host__ __device__ void ProjectHalfspace(Vector<const T> const& v,
                                                 Vector<const T> const& n,
                                                 T t,
                                                 Vector<T>& result,
                                                 int dim)
{
  ProjectHalfspace<T, Vector<T>, Vector<const T>>(v, n, t, result, dim);
}

template<typename T>
inline __host__ __device__ void ProjectHalfspace(const T* const& v,
                                                 const T* const& n,
                                                 T t,
                                                 T* & result,
                                                 int dim)
{
  ProjectHalfspace<T, T*, const T*>(v, n, t, result, dim);
}

template<typename T>
inline __host__ __device__ bool IsPointInHalfspace(
    Vector<const T> const& v,
    Vector<const T> const& p,
    Vector<const T> const& n,
    int dim)
{
  return IsPointInHalfspace<T, Vector<const T>>(v, p, n, dim);
}

template<typename T>
inline __host__ __device__ bool IsPointInHalfspace(
    const T* const& v,
    const T* const& p,
    const T* const& n,
    int dim)
{
  return IsPointInHalfspace<T, const T*>(v, p, n, dim);
}

/// 
/// \brief Computes the singular value decomposition of a 2x2 matrix
///        See: http://ieeexplore.ieee.org/document/486688/
///
///        (Assume "column-first" ordering)
/// 
template<typename T>
inline  __host__ __device__ void computeSVD2x2(const T *a, T *U, T *S, T *V) {
  const T E = (a[0]+a[3]) / 2;
  const T F = (a[0]-a[3]) / 2;
  const T G = (a[2]+a[1]) / 2;
  const T H = (a[2]-a[1]) / 2;

  const T Q = sqrt(E*E+H*H);
  const T R = sqrt(F*F+G*G);

  const T a1 = atan2(G, F);
  const T a2 = atan2(H, E);
  const T beta = (a2-a1)/2;
  const T gamma = (a2+a1)/2;

  V[0] = cos(gamma); V[2] = sin(gamma);
  V[1] = sin(gamma); V[3] = -cos(gamma);

  S[0] = Q + R;
  S[1] = Q - R;

  U[0] = cos(beta);  U[2] = sin(beta);
  U[1] = -sin(beta); U[3] = cos(beta);
}


/// 
/// \brief Multiplies two 4x4 matrices (assume column-first ordering)
///        X = A * B;
/// 
template<typename T>
inline  __host__ __device__ void matMult4(T *X, const T *A, const T *B) {
  X[0] = A[0]*B[0] + A[4]*B[1] + A[8]*B[2] + A[12]*B[3];
  X[1] = A[1]*B[0] + A[5]*B[1] + A[9]*B[2] + A[13]*B[3];
  X[2] = A[2]*B[0] + A[6]*B[1] + A[10]*B[2] + A[14]*B[3];
  X[3] = A[3]*B[0] + A[7]*B[1] + A[11]*B[2] + A[15]*B[3];

  X[4] = A[0]*B[4] + A[4]*B[5] + A[8]*B[6] + A[12]*B[7];
  X[5] = A[1]*B[4] + A[5]*B[5] + A[9]*B[6] + A[13]*B[7];
  X[6] = A[2]*B[4] + A[6]*B[5] + A[10]*B[6] + A[14]*B[7];
  X[7] = A[3]*B[4] + A[7]*B[5] + A[11]*B[6] + A[15]*B[7];

  X[8] = A[0]*B[8] + A[4]*B[9] + A[8]*B[10] + A[12]*B[11];
  X[9] = A[1]*B[8] + A[5]*B[9] + A[9]*B[10] + A[13]*B[11];
  X[10] = A[2]*B[8] + A[6]*B[9] + A[10]*B[10] + A[14]*B[11];
  X[11] = A[3]*B[8] + A[7]*B[9] + A[11]*B[10] + A[15]*B[11];

  X[12] = A[0]*B[12] + A[4]*B[13] + A[8]*B[14] + A[12]*B[15];
  X[13] = A[1]*B[12] + A[5]*B[13] + A[9]*B[14] + A[13]*B[15];
  X[14] = A[2]*B[12] + A[6]*B[13] + A[10]*B[14] + A[14]*B[15];
  X[15] = A[3]*B[12] + A[7]*B[13] + A[11]*B[14] + A[15]*B[15];
}

/// 
/// \brief Multiplies two 4x4 matrices (assume column-first ordering)
///        X = A^T * B;
/// 
template<typename T>
inline  __host__ __device__ void matMult4_AT(T *X, const T *A, const T *B) {
  X[0] = A[0]*B[0] + A[1]*B[1] + A[2]*B[2] + A[3]*B[3];
  X[1] = A[4]*B[0] + A[5]*B[1] + A[6]*B[2] + A[7]*B[3];
  X[2] = A[8]*B[0] + A[9]*B[1] + A[10]*B[2] + A[11]*B[3];
  X[3] = A[12]*B[0] + A[13]*B[1] + A[14]*B[2] + A[15]*B[3];

  X[4] = A[0]*B[4] + A[1]*B[5] + A[2]*B[6] + A[3]*B[7];
  X[5] = A[4]*B[4] + A[5]*B[5] + A[6]*B[6] + A[7]*B[7];
  X[6] = A[8]*B[4] + A[9]*B[5] + A[10]*B[6] + A[11]*B[7];
  X[7] = A[12]*B[4] + A[13]*B[5] + A[14]*B[6] + A[15]*B[7];

  X[8] = A[0]*B[8] + A[1]*B[9] + A[2]*B[10] + A[3]*B[11];
  X[9] = A[4]*B[8] + A[5]*B[9] + A[6]*B[10] + A[7]*B[11];
  X[10] = A[8]*B[8] + A[9]*B[9] + A[10]*B[10] + A[11]*B[11];
  X[11] = A[12]*B[8] + A[13]*B[9] + A[14]*B[10] + A[15]*B[11];

  X[12] = A[0]*B[12] + A[1]*B[13] + A[2]*B[14] + A[3]*B[15];
  X[13] = A[4]*B[12] + A[5]*B[13] + A[6]*B[14] + A[7]*B[15];
  X[14] = A[8]*B[12] + A[9]*B[13] + A[10]*B[14] + A[11]*B[15];
  X[15] = A[12]*B[12] + A[13]*B[13] + A[14]*B[14] + A[15]*B[15];
}

/// 
/// \brief Multiplies two 4x4 matrices (assume column-first ordering)
///        X = A * B^T;
/// 
template<typename T>
inline __host__ __device__ void matMult4_BT(T *X, const T *A, const T *B) {
  X[0] = A[0]*B[0] + A[4]*B[4] + A[8]*B[8] + A[12]*B[12];
  X[1] = A[1]*B[0] + A[5]*B[4] + A[9]*B[8] + A[13]*B[12];
  X[2] = A[2]*B[0] + A[6]*B[4] + A[10]*B[8] + A[14]*B[12];
  X[3] = A[3]*B[0] + A[7]*B[4] + A[11]*B[8] + A[15]*B[12];

  X[4] = A[0]*B[1] + A[4]*B[5] + A[8]*B[9] + A[12]*B[13];
  X[5] = A[1]*B[1] + A[5]*B[5] + A[9]*B[9] + A[13]*B[13];
  X[6] = A[2]*B[1] + A[6]*B[5] + A[10]*B[9] + A[14]*B[13];
  X[7] = A[3]*B[1] + A[7]*B[5] + A[11]*B[9] + A[15]*B[13];

  X[8] = A[0]*B[2] + A[4]*B[6] + A[8]*B[10] + A[12]*B[14];
  X[9] = A[1]*B[2] + A[5]*B[6] + A[9]*B[10] + A[13]*B[14];
  X[10] = A[2]*B[2] + A[6]*B[6] + A[10]*B[10] + A[14]*B[14];
  X[11] = A[3]*B[2] + A[7]*B[6] + A[11]*B[10] + A[15]*B[14];

  X[12] = A[0]*B[3] + A[4]*B[7] + A[8]*B[11] + A[12]*B[15];
  X[13] = A[1]*B[3] + A[5]*B[7] + A[9]*B[11] + A[13]*B[15];
  X[14] = A[2]*B[3] + A[6]*B[7] + A[10]*B[11] + A[14]*B[15];
  X[15] = A[3]*B[3] + A[7]*B[7] + A[11]*B[11] + A[15]*B[15];
}

/// 
/// \brief Reduce a 4x4 skew-symmetric matrix to tridiagonal form
///        A = Q * D * Q';
///
template<typename T>
inline __host__ __device__ void skewReduce4(const T *A, T *Q, T *D) {
  T v[3], f, denom;
  T t1[16], t2[16]; // temporary variables

  v[0] = A[1];
  v[1] = A[2];
  v[2] = A[3];
  v[0] += sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  
  denom = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if(denom > 0)
    f = 2.0 / denom;
  else
    f = 0.;

  t1[0] = 1; t1[4] = 0; t1[8] = 0; t1[12] = 0;
  t1[1] = 0;
  t1[2] = 0;
  t1[3] = 0;
  
  t1[5] = 1. - f * v[0] * v[0]; t1[9] = -f * v[0] * v[1];      t1[13] = -f * v[0] * v[2];
  t1[6] = -f * v[1] * v[0];     t1[10] = 1. - f * v[1] * v[1]; t1[14] = -f * v[1] * v[2];
  t1[7] = -f * v[2] * v[0];     t1[11]= -f * v[2] * v[1];      t1[15] = 1. - f * v[2] * v[2];

  matMult4<T>(t2, A, t1);
  matMult4<T>(D, t1, t2);

  v[0] = D[6];
  v[1] = D[7];
  v[0] += sqrt(v[0] * v[0] + v[1] * v[1]);

  denom = (v[0] * v[0] + v[1] * v[1]);
  if(denom > 0)
    f = 2.0 / denom;
  else
    f = 0.;

  t2[0] = 1; t2[4] = 0; t2[8] = 0; t2[12] = 0;
  t2[1] = 0; t2[5] = 1; t2[9] = 0; t2[13] = 0;
  t2[2] = 0; t2[6] = 0; 
  t2[3] = 0; t2[7] = 0;

  t2[10] = 1 - f * v[0] * v[0]; t2[14] = -f * v[1] * v[0];
  t2[11] = -f * v[0] * v[1];    t2[15] = 1 - f * v[1] * v[1];
 
  matMult4<T>(Q, t2, t1);
  matMult4<T>(t1, D, t2);
  matMult4<T>(D, t2, t1);
}

} // namespace helper
} // namespace prost 

#endif // PROST_HELPER_HPP_
