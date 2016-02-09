#include "prost/prox/prox.hpp"

namespace prost {

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
  const std::vector<T>& arg, 
  const std::vector<T>& tau_diag, 
  T tau) 
{
  const thrust::device_vector<T> d_arg(arg.begin(), arg.end());
  thrust::device_vector<T> d_res;
  d_res.resize(arg.size());
  const thrust::device_vector<T> d_tau(tau_diag.begin(), tau_diag.end());

  Eval(d_res, d_arg, d_tau, tau);

  result.resize(arg.size());
  thrust::copy(d_res.begin(), d_res.end(), result.begin());
}

// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;

} // namespace prost
