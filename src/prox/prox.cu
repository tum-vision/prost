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

template<typename T>
void Prox<T>::Eval(std::vector<T>& arg, std::vector<T>& res, std::vector<T>& tau_diag, T tau) {
    thrust::device_vector<T> d_arg(arg.begin(), arg.end());
    thrust::device_vector<T> d_res;
    d_res.resize(arg.size());
    thrust::device_vector<T> d_tau(tau_diag.begin(), tau_diag.end());

    Eval(d_arg, d_res, d_tau, tau);

    thrust::copy(d_res.begin(), d_res.end(), res.begin());
}

// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;
