#ifndef PROX_EPI_PIECEW_LIN_HPP_
#define PROX_EPI_PIECEW_LIN_HPP_

#include <vector>

#include "prox.hpp"

template<typename T>
struct EpiPiecewLinCoeffs {
    std::vector<T> x, y;
    std::vector<T> alpha, beta;
    std::vector<size_t> count;
    std::vector<size_t> index;    
};

template<typename T>
struct EpiPiecewLinCoeffsDevice {
    T *d_ptr_x;
    T *d_ptr_y;
    T *d_ptr_alpha;
    T *d_ptr_beta;
    size_t *d_ptr_count;
    size_t *d_ptr_index;
};

/**
 * @brief Computes orthogonal projection of (u,v) onto the convex set
 *        C = { (x, y) | y >= (ax^2 + bx + c + delta(alpha <= x <= beta))* },
 *        where * denotes the Legendre-Fenchel conjugate.
 */
template<typename T>
class ProxEpiPiecewLin : public Prox<T> {
 public:
  ProxEpiPiecewLin(size_t index,
                   size_t count,
                   bool interleaved,
                   const EpiPiecewLinCoeffs<T>& coeffs);

  virtual ~ProxEpiPiecewLin();

  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
  
  EpiPiecewLinCoeffs<T> coeffs_;
  EpiPiecewLinCoeffsDevice<T> coeffs_dev_;
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
