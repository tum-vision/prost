#ifndef PROX_EPI_PARABOLA_HPP_
#define PROX_EPI_PARABOLA_HPP_

#include <vector>
#include "prox.hpp"

/**
 * @brief Computes orthogonal projection of (phi^x,phi^t) onto the convex set
 *        C = { (phi^x, phi^t) | phi^t + g >= alpha * ||phi^x||^2 }
 */
template<typename T>
class ProxEpiParabola : public Prox<T> {
public:
  ProxEpiParabola(
    size_t index,
    size_t count,
    size_t dim,
    const std::vector<T>& g,
    T alpha);

  virtual ~ProxEpiParabola();

  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau);
  
  T *d_g_;
  std::vector<T> g_;
  T alpha_;
};

#ifdef __CUDACC__

/**
 * @brief Computes the orthogonal projection of (x0, y0) onto the epigraph of
 *        the parabola y >= \alpha ||x||^2.
 */
template<typename T>
inline __device__ 
void ProjectParabolaSimpleNd(
  const T* x0, const T& y0, const T& alpha, T* x, T& y, size_t dim)
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

/**
 * @brief Computes orthogonal projection of (x0, y0) onto the epigraph of the
 *        parabola y + g >= alpha * x^2.
 */
template<typename T>
inline __device__ void 
ProjectParabolaShiftedNd(
  T* x0, const T& y0, const T& alpha, const T& g, T* x, T& y, size_t dim)
{
  ProjectParabolaSimpleNd(x0, y0 + g, alpha, x, y, dim);
  y -= g;
}

#endif
#endif


