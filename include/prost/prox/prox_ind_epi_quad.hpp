#ifndef PROST_PROX_IND_EPI_QUAD_
#define PROST_PROX_IND_EPI_QUAD_

#include <array>
#include <vector>
#include <thrust/device_vector.h>

#include "prost/prox/prox_separable_sum.hpp"
#include "prost/prox/vector.hpp"
#include "prost/common.hpp"

namespace prost {

///
/// \brief Implements projection onto epigraph of quadratic function s.t. a/2 x^T x + b^T x + c >= y
///
template<typename T>
class ProxIndEpiQuad : public ProxSeparableSum<T> 
{
public:    
  ProxIndEpiQuad(
      size_t index, 
      size_t count, 
      size_t dim, 
      bool interleaved, 
      bool diagsteps, 
      std::vector<T> a,
      std::vector<T> b,
      std::vector<T> c)
      : ProxSeparableSum<T>(index, count, dim, interleaved, diagsteps), a_(a), b_(b), c_(c) { }

  virtual void Initialize();
  
  virtual size_t gpu_mem_amount() const
  {
    return (this->count_*this->dim_ + this->count_) *sizeof(T);
  }
   
protected:
  virtual void EvalLocal(
    const typename thrust::device_vector<T>::iterator& result_beg,
    const typename thrust::device_vector<T>::iterator& result_end,
    const typename thrust::device_vector<T>::const_iterator& arg_beg,
    const typename thrust::device_vector<T>::const_iterator& arg_end,
    const typename thrust::device_vector<T>::const_iterator& tau_beg,
    const typename thrust::device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);
  
private:    
  thrust::device_vector<T> d_a_;
  thrust::device_vector<T> d_b_;
  thrust::device_vector<T> d_c_;  
 
  std::vector<T> a_;
  std::vector<T> b_;
  std::vector<T> c_;
};

} // namespace prost

#endif // PROST_PROJ_EPI_QUAD_
