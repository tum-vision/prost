#include "prox.hpp"

template<typename T>
void Prox<T>::Eval(T *d_arg, T *d_result, T tau, T *d_tau) {
  EvalLocal(&d_arg[index_],
            &d_result[index_],
            tau,
            d_tau[index_]);
}

// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;
