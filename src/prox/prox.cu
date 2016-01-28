#include "prox/prox.hpp"

template<typename T>
void Prox<T>::Eval(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& arg, 
  const thrust::device_vector<T>& tau_diag, 
  T tau)
{
  EvalLocal(
    result.begin() + index_,
    result.begin() + index_ + size_,
    arg.cbegin() + index_,
    arg.cbegin() + index_ + size_,
    tau_diag.cbegin() + index_,
    tau_diag.cbegin() + index_ + size_,
    tau,
    false);
}

template<typename T>
void Prox<T>::Eval(
  std::vector<T>& result, 
  std::vector<T>& arg, 
  std::vector<T>& tau_diag, 
  T tau) 
{
  thrust::device_vector<T> d_arg(arg.begin(), arg.end());
  thrust::device_vector<T> d_res;
  d_res.resize(arg.size());
  thrust::device_vector<T> d_tau(tau_diag.begin(), tau_diag.end());

  Eval(d_arg, d_res, d_tau, tau);

  result.resize(arg.size());
  thrust::copy(d_res.begin(), d_res.end(), result.begin());
}

// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;
