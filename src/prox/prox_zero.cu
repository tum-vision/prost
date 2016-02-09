#include "prost/prox/prox_zero.hpp"

namespace prost {

template<typename T>
ProxZero<T>::ProxZero(size_t index, size_t size) :
    Prox<T>(index, size, true)
{
}

template<typename T>
ProxZero<T>::~ProxZero() 
{
}

template<typename T>
void ProxZero<T>::EvalLocal(
  const typename thrust::device_vector<T>::iterator& result_beg,
  const typename thrust::device_vector<T>::iterator& result_end,
  const typename thrust::device_vector<T>::const_iterator& arg_beg,
  const typename thrust::device_vector<T>::const_iterator& arg_end,
  const typename thrust::device_vector<T>::const_iterator& tau_beg,
  const typename thrust::device_vector<T>::const_iterator& tau_end,
  T tau,
  bool invert_tau)
{
  thrust::copy(arg_beg, arg_end, result_beg);
}

// Explicit template instantiation
template class ProxZero<float>;
template class ProxZero<double>;

} // namespace prost