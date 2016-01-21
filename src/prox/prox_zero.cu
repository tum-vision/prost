#include "prox/prox_zero.hpp"

using namespace prox;

template<typename T>
ProxZero<T>::ProxZero(size_t index, size_t size) :
    Prox<T>(index, size, true) {
}

template<typename T>
void ProxZero<T>::EvalLocal(typename thrust::device_vector<T>::iterator d_arg_begin,
                         typename thrust::device_vector<T>::iterator d_arg_end,
                         typename thrust::device_vector<T>::iterator d_res_begin,
                         typename thrust::device_vector<T>::iterator d_res_end,
                         typename thrust::device_vector<T>::iterator d_tau_begin,
                         typename thrust::device_vector<T>::iterator d_tau_end,
                         T tau,
                         bool invert_tau) {
  thrust::copy(d_arg_begin, d_arg_end,
             d_res_begin);
}

template<typename T>
size_t ProxZero<T>::gpu_mem_amount() {
    return 0;
}

// Explicit template instantiation
template class ProxZero<float>;
template class ProxZero<double>;
