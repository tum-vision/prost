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

 protected:
  virtual void EvalLocal(device_vector<T> d_arg,
                         device_vector<T> d_res,
                         device_vector<T> d_tau,
                         T tau,
                         bool invert_tau);
};

#endif
