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
    const thrust::device_ptr<T>& result,
    const thrust::device_ptr<const T>& arg,
    const thrust::device_ptr<const T>& tau_diag,
    T tau_scal,
    bool invert_tau) = 0;
};

#endif
