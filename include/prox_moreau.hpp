#ifndef PROX_MOREAU_HPP_
#define PROX_MOREAU_HPP_

#include "prox.hpp"

/**
 * @brief Evaluates the conjugate prox using Moreau's identity.
 *
 */
class ProxMoreau : public Prox {
public:
  ProxMoreau(Prox *conjugate);
  virtual ~ProxMoreau();

  virtual void Evaluate(
      real *d_proxarg,
      real *d_result,
      real tau,
      real *d_tau,
      bool invert_tau = false);

protected:
  Prox *conjugate_;
};

#endif
