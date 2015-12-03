#ifndef PROX_EPI_HUBER_HPP_
#define PROX_EPI_HUBER_HPP_

#include <vector>

#include "prox.hpp"
#include "prox_epi_parabola.hpp"

/**
 * @brief Computes orthogonal projection of (phi^x, phi^t) onto the convex set
 *        C = { (phi^x, phi^t) | phi^t + g >= alpha * |phi^x|^2,  |phi^x| <= 1 }
 */
template<typename T>
class ProxEpiHuber : public Prox<T> {
public:
  ProxEpiHuber(
    size_t index,
    size_t count,
    size_t dim,
    const std::vector<T>& g,
    T alpha);

  virtual ~ProxEpiHuber();

  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau);
  
  T *d_g_;
  std::vector<T> g_;
  T alpha_;
};

#endif
