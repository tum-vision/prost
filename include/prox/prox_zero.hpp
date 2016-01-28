#ifndef PROX_ZERO_HPP_
#define PROX_ZERO_HPP_

#include "prox.hpp"

/**
 * @brief Prox of zero function, (just copies input argument).
 * 
 */
template<typename T>
class ProxZero : public Prox<T> {
public:
  ProxZero(size_t index, size_t size);
  virtual ~ProxZero();

  virtual size_t gpu_mem_amount() const { return 0; }

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

#endif
