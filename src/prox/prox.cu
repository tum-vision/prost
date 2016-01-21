#include "prox/prox.hpp"

template<typename T>
void Prox<T>::Eval(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& arg, 
  const thrust::device_vector<T>& tau_diag, 
  T tau_scal)
{
  EvalLocal(
    &result[index_],
    &arg[index_],
    &tau_diag[index_],
    tau_scal,
    false);
}

// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;
