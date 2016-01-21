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
  virtual void EvalLocal(typename thrust::device_vector<T>::iterator d_arg_begin,
                         typename thrust::device_vector<T>::iterator d_arg_end,
                         typename thrust::device_vector<T>::iterator d_res_begin,
                         typename thrust::device_vector<T>::iterator d_res_end,
                         typename thrust::device_vector<T>::iterator d_tau_begin,
                         typename thrust::device_vector<T>::iterator d_tau_end,
                         T tau,
                         bool invert_tau);
};
}
#endif
