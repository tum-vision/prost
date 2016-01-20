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
void ProxZero<T>::EvalLocal(device_vector<T> d_arg,
                            device_vector<T> d_res,
                            device_vector<T> d_tau,
                            T tau,
                            bool invert_tau) {
  copy(d_arg.begin(), d_arg.end(),
             d_res.begin());
}

// Explicit template instantiation
template class ProxZero<float>;
template class ProxZero<double>;
