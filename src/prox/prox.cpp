#include "prox/prox.hpp"

using namespace prox;

template<typename T>
Prox<T>::~Prox() {
    Release();
}

template<typename T>
void Prox<T>::Eval(thrust::device_vector<T>& arg, thrust::device_vector<T>& res, thrust::device_vector<T>& tau_diag, T tau) {
  EvalLocal(arg.begin() + index_,
            arg.begin() + index_ + size_,
            res.begin() + index_,
            res.begin() + index_ + size_,
            tau_diag.begin() + index_,
            tau_diag.begin() + index_ + size_,
            tau,
            false);
}



// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;
