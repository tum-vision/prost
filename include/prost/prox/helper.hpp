#ifndef PROST_HELPER_HPP_
#define PROST_HELPER_HPP_

#include "prost/prox/vector.hpp"

namespace prost {

/// \brief Namspace grouping prox helper functions.
namespace helper {

///
/// \brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
///        the parabola y >= \alpha ||x||^2 with \alpha > 0.
///
/// 
template<typename T>
inline __host__ __device__ void  ProjectEpiQuadNd(
     Vector<T>& x0, const T y0, const T alpha, Vector<T>& x, T& y, size_t dim)
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
/// \brief Computes projection of the d-dimensional vector v onto the
///        halfspace described by { x | <n, x> <= t }.
/// 
template<typename T>
inline __device__ void ProjectHalfspace(
  const Vector<const T>& v,
  const Vector<const T>& n,
  T t,
  Vector<T>& result,
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
template<typename T>
inline __device__ bool IsPointInHalfspace(
  const Vector<const T>& v,
  const Vector<const T>& p,
  const Vector<const T>& n,
  int dim)
{
  T dot = 0;
  for(int i = 0; i < dim; i++) 
    dot += n[i] * (v[i] - p[i]);
  
  return dot <= 0.;
}

} // namespace helper
} // namespace prost 

#endif // PROST_HELPER_HPP_
