#ifndef PROX_MOREAU_HPP_
#define PROX_MOREAU_HPP_

#include "prox.hpp"

#include <memory>

#include <thrust/device_vector.h>

/**
 * @brief Evaluates the conjugate prox using Moreau's identity.
 *
 */
template<typename T>
class ProxMoreau : public Prox<T> {
public:
  ProxMoreau(std::shared_ptr<Prox<T> > conjugate);
  virtual ~ProxMoreau();

  virtual void Initialize();
  virtual void Release();

  virtual size_t gpu_mem_amount() const;

protected:
  std::shared_ptr<Prox<T> > conjugate_;
  thrust::device_vector<T> scaled_arg_;

  virtual void EvalLocal(
    const typename thrust::device_vector<T>::iterator& result_beg,
    const typename thrust::device_vector<T>::iterator& result_end,
    const typename thrust::device_vector<T>::const_iterator& arg_beg,
    const typename thrust::device_vector<T>::const_iterator& arg_end,
    const typename thrust::device_vector<T>::const_iterator& tau_beg,
    const typename thrust::device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);
};

#endif
