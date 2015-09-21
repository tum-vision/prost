#include "prox/prox.hpp"

template<typename T>
void Prox<T>::Eval(T *d_arg, T *d_result, T* d_tau, T tau) {
  EvalLocal(&d_arg[index_],
            &d_result[index_],
            &d_tau[index_],
            tau,
            false);
}

// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;
