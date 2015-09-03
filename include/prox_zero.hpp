#ifndef PROX_ZERO_HPP_
#define PROX_ZERO_HPP_

#include "prox.hpp"

/**
 * @brief Prox of zero function, copies input argument.
 * 
 */
class ProxZero : public Prox {
public:
  ProxZero(int index, int count);
  virtual ~ProxZero();

  virtual void Evaluate(
      real *d_arg,
      real *d_result,
      real tau,
      real *d_tau,
      bool invert_step = false);
};

#endif
