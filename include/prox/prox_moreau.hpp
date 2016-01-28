#ifndef PROX_MOREAU_HPP_
#define PROX_MOREAU_HPP_

#include "prox.hpp"

#include <memory>

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

/**
 * @brief Evaluates the conjugate prox using Moreau's identity.
 *
 */
template<typename T>
class ProxMoreau : public Prox<T> {
public:
  ProxMoreau(std::shared_ptr<Prox<T> > conjugate);
  virtual ~ProxMoreau();

  virtual void Init();
  virtual void Release();

  virtual size_t gpu_mem_amount() const;

protected:
  std::shared_ptr<Prox<T> > conjugate_;
  thrust::device_vector<T> scaled_arg_;

  virtual void EvalLocal(
    const thrust::device_ptr<T>& result,
    const thrust::device_ptr<const T>& arg,
    const thrust::device_ptr<const T>& tau_diag,
    T tau_scal,
    bool invert_tau);
};

#endif
