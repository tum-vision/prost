#ifndef ELEM_OPERATION_HALFSPACE_HPP_
#define ELEM_OPERATION_HALFSPACE_HPP_

#include <vector>

#include "prox.hpp"

/**
 * @brief Computes orthogonal projection of (u,v) onto the convex set
 *        C = { (x, y) | <x,a> <= 0 }.
 */
template<typename T>
class ProxHalfspace : public Prox<T> {
 public:
  ProxHalfspace(size_t index,
                size_t count,
                size_t dim,
                bool interleaved,         
                std::vector<T>& a);

  virtual ~ProxHalfspace();

  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
  
  std::vector<T> a_;
  T* d_ptr_a_;
};

#endif
