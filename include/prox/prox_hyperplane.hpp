#ifndef PROX_HYPERPLANE_HPP_
#define PROX_HYPERPLANE_HPP_

#include <vector>

#include "prox.hpp"

/**
 * @brief Computes orthogonal projection of (u,v) onto the convex set
 *        C = { (x, y) | <x,b> <= y }.
 */
template<typename T>
class ProxHyperplane : public Prox<T> {
 public:
  ProxHyperplane(size_t index,
                   size_t count,
                   size_t dim,
                   std::vector<T>& b);

  virtual ~ProxHyperplane();

  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
  
  std::vector<T> b_;
  T* d_ptr_b_;  
};

#ifdef __CUDACC__

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
