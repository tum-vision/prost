#ifndef PROST_PROX_IND_EPI_POLYHEDRAL_1D_HPP_
#define PROST_PROX_IND_EPI_POLYHEDRAL_1D_HPP_

#include "prost/prox/prox_separable_sum.hpp"

namespace prost {

/// 
/// \brief Computes orthogonal projection onto the epigraph
///        of a one-dimensional piecewise linear function.
/// 
///        TODO: comment what is pt_x, pt_y, alpha, beta, count and index.
/// 
class ProxIndEpiPolyhedral1D : public ProxSeparableSum<T> {
public:
  ProxIndEpiPolyhedral1D(
    size_t index,
    size_t count,
    bool interleaved,
    const vector<T>& pt_x, 
    const vector<T>& pt_y,
    const vector<T>& alpha,
    const vector<T>& beta,
    const vector<size_t>& count,
    const vector<size_t>& index);

  virtual ~ProxIndEpiPolyhedral1D() { }

  virtual void Initialize();
  virtual size_t gpu_mem_amount() const;

protected:
  virtual void EvalLocal(
    const typename device_vector<T>::iterator& result_beg,
    const typename device_vector<T>::iterator& result_end,
    const typename device_vector<T>::const_iterator& arg_beg,
    const typename device_vector<T>::const_iterator& arg_end,
    const typename device_vector<T>::const_iterator& tau_beg,
    const typename device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);
  
private:
  vector<T> host_pt_x_, host_pt_y_, host_alpha_, host_beta_;
  vector<size_t> host_count_, host_index_;

  device_vector<T> dev_pt_x_, dev_pt_y_, dev_alpha_, dev_beta_;
  device_vector<size_t> dev_count_, dev_index_;
};

} // namespace prost

#endif // PROST_PROX_IND_EPI_POLYHEDRAL_1D_HPP_

/*
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

/// 
/// \brief Computes orthogonal projection of (u,v) onto the convex set
///        C = { (x, y) | y >= (ax^2 + bx + c + delta(alpha <= x <= beta))* },
///        where * denotes the Legendre-Fenchel conjugate.
///
template<typename T>
class ProxEpiPiecewLin : public Prox<T> {
 public:
  ProxEpiPiecewLin(
    size_t index,
    size_t count,
    bool interleaved,
    const EpiPiecewLinCoeffs<T>& coeffs,
    T scaling = 1);

  virtual ~ProxEpiPiecewLin();

  virtual size_t gpu_mem_amount(); 

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
  T scaling_;
};

#ifdef __CUDACC__

/// 
/// \brief Computes projection of the d-dimensional vector v onto the
///        halfspace described by { x | <n, x> <= t }.
/// 
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

/// 
/// \brief Checks whether a d-dimensional point v lies within the halfspace
///        described by a point p and normal n.
///
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
*/
