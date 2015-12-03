#ifndef PROX_PLUS_LINTERM_HPP_
#define PROX_PLUS_LINTERM_HPP_

#include "prox.hpp"

#include <vector>

/**
 * @brief Evalutes the proximal operator of f(x) + <c,x> by applying the 
 *        proximal operator of f to a shifted version of x0.
 * 
 */
template<typename T>
class ProxPlusLinterm : public Prox<T> {
public:
  ProxPlusLinterm(
    Prox<T> *prox,
    const std::vector<T>& c
    ); 
  virtual ~ProxPlusLinterm();

  virtual bool Init();
  virtual void Release();

  virtual size_t gpu_mem_amount();

protected:
  Prox<T> *prox_;
  std::vector<T> c_;
  T *d_c_;

  virtual void EvalLocal(
    T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau);
};

#endif
