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
  ProxZero(size_t index, size_t count);
  virtual ~ProxZero();

 protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
};

#endif
