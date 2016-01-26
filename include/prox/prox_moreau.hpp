#ifndef PROX_MOREAU_HPP_
#define PROX_MOREAU_HPP_

#include "prox.hpp"

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include "../pdsolver_exception.hpp"


/**
 * @brief Evaluates the conjugate prox using Moreau's identity.
 *
 */
namespace prox {
template<typename T>
class ProxMoreau : public prox::Prox<T> {
public:
  ProxMoreau(std::unique_ptr<prox::Prox<T>> conjugate);
  virtual ~ProxMoreau();

  virtual void Init();

  virtual size_t gpu_mem_amount();

protected:
  std::unique_ptr<prox::Prox<T>> conjugate_;
  thrust::device_vector<T> d_scaled_arg_;

  virtual void EvalLocal(typename thrust::device_vector<T>::iterator d_arg_begin,
                         typename thrust::device_vector<T>::iterator d_arg_end,
                         typename thrust::device_vector<T>::iterator d_res_begin,
                         typename thrust::device_vector<T>::iterator d_res_end,
                         typename thrust::device_vector<T>::iterator d_tau_begin,
                         typename thrust::device_vector<T>::iterator d_tau_end,
                         T tau,
                         bool invert_tau);

};
}
#endif
