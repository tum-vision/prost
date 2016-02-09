#ifndef PROST_PROX_MOREAU_HPP_
#define PROST_PROX_MOREAU_HPP_

#include <thrust/device_vector.h>

#include "prost/prox/prox.hpp"
#include "prost/common.hpp"

namespace prost {

/// 
/// \brief Evaluates the conjugate prox using Moreau's identity.
/// 
template<typename T>
class ProxMoreau : public Prox<T> {
public:
  ProxMoreau(shared_ptr<Prox<T>> conjugate);
  virtual ~ProxMoreau();

  virtual void Initialize();
  virtual void Release();

  virtual size_t gpu_mem_amount() const;

protected:
  shared_ptr<Prox<T>> conjugate_;
  device_vector<T> scaled_arg_;

  virtual void EvalLocal(
    const typename device_vector<T>::iterator& result_beg,
    const typename device_vector<T>::iterator& result_end,
    const typename device_vector<T>::const_iterator& arg_beg,
    const typename device_vector<T>::const_iterator& arg_end,
    const typename device_vector<T>::const_iterator& tau_beg,
    const typename device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);
};

} // namespace prost

#endif // PROST_PROX_MOREAU_HPP_
