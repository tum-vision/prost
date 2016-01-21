#include "prox/prox_zero.hpp"

template<typename T>
ProxZero<T>::ProxZero(size_t index, size_t size) :
    Prox<T>(index, size, true)
{
}

template<typename T>
ProxZero<T>::~ProxZero() {
}

template<typename T>
void ProxZero<T>::EvalLocal(
  const thrust::device_ptr<T>& result,
  const thrust::device_ptr<const T>& arg,
  const thrust::device_ptr<const T>& tau_diag,
  T tau_scal,
  bool invert_tau)
{
//  thrust::copy(arg, arg + this->size_, result);
}

// Explicit template instantiation
template class ProxZero<float>;
template class ProxZero<double>;
