#include "prox/prox_zero.hpp"

using namespace prox;

template<typename T>
ProxZero<T>::ProxZero(size_t index, size_t size) :
    Prox<T>(index, size, true) {
}

template<typename T>
void ProxZero<T>::EvalLocal(const typename thrust::device_vector<T>::iterator& arg_begin,
                         const typename thrust::device_vector<T>::iterator& arg_end,
                         const typename thrust::device_vector<T>::iterator& res_begin,
                         const typename thrust::device_vector<T>::iterator& res_end,
                         const typename thrust::device_vector<T>::iterator& tau_begin,
                         const typename thrust::device_vector<T>::iterator& tau_end,
                         T tau,
                         bool invert_tau) {
  thrust::copy(arg_begin, arg_end,
             res_begin);
}

template<typename T>
size_t ProxZero<T>::gpu_mem_amount() {
    return 0;
}

// Explicit template instantiation
template class ProxZero<float>;
template class ProxZero<double>;
