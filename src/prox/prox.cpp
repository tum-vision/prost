#include "prox/prox.hpp"

using namespace prox;

template<typename T>
Prox<T>::~Prox() {
    Release();
}

template<typename T>
void Prox<T>::Eval(thrust::device_vector<T> d_arg, thrust::device_vector<T> d_res, thrust::device_vector<T> d_tau, T tau) {
  EvalLocal(d_arg.begin() + index_,
            d_arg.begin() + index_ + size_,
            d_res.begin() + index_,
            d_res.begin() + index_ + size_,
            d_tau.begin() + index_,
            d_tau.begin() + index_ + size_,
            tau,
            false);
}

// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;
