#include "prox/prox_zero.hpp"

template<typename T>
ProxZero<T>::ProxZero(size_t index, size_t count) :
    Prox<T>(index, count, 1, false, true)
{
}

template<typename T>
ProxZero<T>::~ProxZero() {
}

template<typename T>
void ProxZero<T>::EvalLocal(T *d_arg,
                            T *d_res,
                            T *d_tau,
                            T tau,
                            bool invert_tau)
{
  cudaMemcpy(d_res,
             d_arg,
             sizeof(T) * this->count_,
             cudaMemcpyDeviceToDevice);
}

// Explicit template instantiation
template class ProxZero<float>;
template class ProxZero<double>;
