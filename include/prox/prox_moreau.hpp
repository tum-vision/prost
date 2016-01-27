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

  virtual void Init();

  virtual size_t gpu_mem_amount();

protected:
  std::unique_ptr<prox::Prox<T>> conjugate_;
  thrust::device_vector<T> scaled_arg_;

  virtual void EvalLocal(const typename thrust::device_vector<T>::iterator& arg_begin,
                         const typename thrust::device_vector<T>::iterator& arg_end,
                         const typename thrust::device_vector<T>::iterator& res_begin,
                         const typename thrust::device_vector<T>::iterator& res_end,
                         const typename thrust::device_vector<T>::iterator& tau_begin,
                         const typename thrust::device_vector<T>::iterator& tau_end,
                         T tau,
                         bool invert_tau);

};
}
#endif
