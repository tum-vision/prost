#ifndef PROST_MATH_FUNCTIONS_HPP_
#define PROST_MATH_FUNCTIONS_HPP_

namespace prost {

///
/// \brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
///        the parabola y >= \alpha x^2.
/// 
template<typename T>
inline __host__ __device__ void ProjectParabolaSimple(
  const T& x0,
  const T& y0,
  const T& alpha,
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
inline __host__ __device__ void ProjectParabolaGeneral(
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
  
  ProjectParabolaSimple<T>(
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
///        halfspace described by { x | <n, x> <= t }.
/// 
template<typename T>
inline __device__ void ProjectHalfspace(
  const T *v,
  const T *n,
  const T& t,
  T* result,
  int d)
{
  T dot = 0, sq_norm = 0;

  for(int i = 0; i < d; i++) {
    dot += n[i] * v[i];
    sq_norm += n[i] * n[i];
  }

  const T s = (dot - t) / sq_norm;
  
  for(int i = 0; i < d; i++) 
    result[i] = v[i] - s * n[i];
}

/// 
/// \brief Checks whether a d-dimensional point v lies within the halfspace
///        described by a point p and normal n.
/// 
template<typename T>
inline __device__ bool PointInHalfspace(
  const T* v,
  const T* p,
  const T* n,
  int d)
{
  T dot = 0;
  for(int i = 0; i < d; i++) 
    dot += n[i] * (v[i] - p[i]);
  
  return dot <= 0.;
}

} // namespace prost 

#endif // PROST_MATH_FUNCTIONS_HPP_
