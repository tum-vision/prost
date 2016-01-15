#ifndef PROX_MOREAU_HPP_
#define PROX_MOREAU_HPP_

#include "prox.hpp"

/**
 * @brief Evaluates the conjugate prox using Moreau's identity.
 *
 */
template<typename T>
class ProxMoreau : public Prox<T> {
public:
  ProxMoreau(Prox<T> *conjugate);
  virtual ~ProxMoreau();

  virtual bool Init();
  virtual void Release();

  virtual size_t gpu_mem_amount();

protected:
  Prox<T> *conjugate_;
  T *d_scaled_arg_;

  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);

};

#endif
