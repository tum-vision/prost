#ifndef PROX_EPI_CONJQUADR_HPP_
#define PROX_EPI_CONJQUADR_HPP_

#include <vector>

#include "prox.hpp"

#define PROX_EPI_CONJQUADR_NUM_COEFFS 5

template<typename T>
struct EpiConjQuadrCoeffs {
  std::vector<T> a, b, c;
  std::vector<T> alpha, beta;
};

template<typename T>
struct EpiConjQuadrCoeffsDevice {
  T *d_ptr[PROX_EPI_CONJQUADR_NUM_COEFFS];
};

/**
 * @brief Computes orthogonal projection of (u,v) onto the convex set
 *        C = { (x, y) | y >= (ax^2 + bx + c + delta(alpha <= x <= beta))* },
 *        where * denotes the Legendre-Fenchel conjugate.
 */
template<typename T>
class ProxEpiConjQuadr : public Prox<T> {
 public:
  ProxEpiConjQuadr(size_t index,
                   size_t count,
                   bool interleaved,
                   const EpiConjQuadrCoeffs<T>& coeffs);

  virtual ~ProxEpiConjQuadr();

  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
  
  EpiConjQuadrCoeffs<T> coeffs_;
  EpiConjQuadrCoeffsDevice<T> coeffs_dev_;
};

#ifdef __CUDACC__

/**
 * @brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
 *        the parabola y >= \alpha x^2.
 */
template<typename T>
inline __device__ void ProjectParabolaSimple(const T& x0,
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
    const T a = 2. * alpha * (x0 * x0);
    const T b = 2. * (1. - 2. * alpha * y0) / 3.;
    const T sq = pow(-b, 3. / 2.);
    const T d = (a - sq) * (a + sq);

    T v;
    if(d >= 0) {
      const T c = pow(a + sqrt(d), 1. / 3.);
      v = c - b / c;
    }
    else {
      v = 2 * sqrt(-b) * cos(acos(a / pow(-b, 3. / 2.)) / 3.);
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

/**
 * @brief Computes orthogonal projection of (x0, y0) onto the epigraph of the
 *        parabola y >= p * x^2 + q * x + r.
 */
template<typename T>
inline __device__ void ProjectParabolaGeneral(const T& x0,
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

/**
 * @brief Computes projection of the d-dimensional vector v onto the
 *        halfspace described by { x | <n, x> <= t }.
 */
template<typename T>
inline __device__ void ProjectHalfspace(const T *v,
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

/**
 * @brief Checks whether a d-dimensional point v lies within the halfspace
 *        described by a point p and normal n.
 */
template<typename T>
inline __device__ bool PointInHalfspace(const T* v,
                                        const T* p,
                                        const T* n,
                                        int d)
{
  T dot = 0;
  for(int i = 0; i < d; i++) 
    dot += n[i] * (v[i] - p[i]);
  
  return dot <= 0.;
}

#endif

#endif
