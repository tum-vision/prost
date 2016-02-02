#ifndef PROX_ELEM_OPERATION_HPP_
#define PROX_ELEM_OPERATION_HPP_

#include <stddef.h>

#include "prox/prox_separable_sum.hpp"

namespace prox 
{

template<typename T, class ELEM_OPERATION, class ENABLE = void>
class ProxElemOperation {};

template<typename T, class ELEM_OPERATION>
class ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::coeffs_count == 0>::type> : public ProxSeparableSum<T> 
{
public:    
  ProxElemOperation(size_t index, size_t count, size_t dim, bool interleaved, bool diagsteps) 
    : ProxSeparableSum<T>(index, count, ELEM_OPERATION::dim <= 0 ? dim : ELEM_OPERATION::dim, interleaved, diagsteps) {}
  
  // set/get methods
  virtual size_t gpu_mem_amount() {
    return 0;
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

};

template<typename T, class ELEM_OPERATION>
class ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::coeffs_count != 0>::type> : public ProxSeparableSum<T> 
{
public:    
  ProxElemOperation(size_t index, 
          size_t count, 
          size_t dim, 
          bool interleaved, 
          bool diagsteps, 
          std::array<std::vector<T>, ELEM_OPERATION::coeffs_count> coeffs) :
      ProxSeparableSum<T>(index, count, ELEM_OPERATION::dim <= 0 ? dim : ELEM_OPERATION::dim, interleaved, diagsteps), 
              coeffs_(coeffs) {}

  virtual void Initialize();
  
  // set/get methods
  virtual size_t gpu_mem_amount() {
    size_t mem = 0;
    for(size_t i = 0; i < ELEM_OPERATION::coeffs_count; i++) {
        if(coeffs_[i].size() > 1)
            mem += this->count_ * sizeof(T);
    }
    return mem;
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
  std::array<std::vector<T>, ELEM_OPERATION::kCoeffsCount> coeffs_;
  std::array<thrust::device_vector<T>, ELEM_OPERATION::kCoeffsCount> d_coeffs_;  
};
}
#endif
