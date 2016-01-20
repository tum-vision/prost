#ifndef PROX_MOREAU_HPP_
#define PROX_MOREAU_HPP_

#include "prox.hpp"

#include <thrust/for_each.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

using namespace thrust;
using namespace std;

/**
 * @brief Evaluates the conjugate prox using Moreau's identity.
 *
 */
template<typename T>
class ProxMoreau : public Prox<T> {
public:
  ProxMoreau(unique_ptr<Prox<T>> conjugate);
  virtual ~ProxMoreau();

  virtual bool Init();
  virtual void Release();

  virtual size_t gpu_mem_amount();

protected:
  unique_ptr<Prox<T>> conjugate_;
  device_vector<T> d_scaled_arg_;

  virtual void EvalLocal(device_vector<T> d_arg,
                         device_vector<T> d_res,
                         device_vector<T> d_tau,
                         T tau,
                         bool invert_tau);

};

#endif
