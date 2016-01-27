#ifndef PROX_ZERO_HPP_
#define PROX_ZERO_HPP_

#include "prox.hpp"



/**
 * @brief Prox of zero function, (just copies input argument).
 * 
 */
namespace prox {
template<typename T>
class ProxZero : public Prox<T> {
 public:
  ProxZero(size_t index, size_t size);

 protected:
  virtual size_t gpu_mem_amount();
  virtual void EvalLocal(const typename thrust::device_vector<T>::iterator& arg_begin,
                         const typename thrust::device_vector<T>::iterator& arg_end,
                         const typename thrust::device_vector<T>::iterator& res_begin,
                         const typename thrust::device_vector<T>::iterator& res_end,
                         const typename thrust::device_vector<T>::iterator& tau_begin,
                         const typename thrust::device_vector<T>::iterator& tau_end,
                         T tau,
                         bool invert_tau);
};
}
#endif
