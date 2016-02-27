#include "prost/prox/prox.hpp"
#include <ctime>

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
double Prox<T>::Eval(
  std::vector<T>& result, 
  const std::vector<T>& arg, 
  const std::vector<T>& tau_diag, 
  T tau) 
{
  const int repeats = 1;

  const thrust::device_vector<T> d_arg(arg.begin(), arg.end());
  thrust::device_vector<T> d_res;
  d_res.resize(arg.size());
  const thrust::device_vector<T> d_tau(tau_diag.begin(), tau_diag.end());

  const clock_t begin_time = clock();
  for(int i = 0; i < repeats; i++)
  {
    Eval(d_res, d_arg, d_tau, tau);
    cudaDeviceSynchronize();
  }
  double s = (double)(clock() - begin_time) / CLOCKS_PER_SEC;

  result.resize(arg.size());
  thrust::copy(d_res.begin(), d_res.end(), result.begin());

  return (s * 1000 / (double)repeats);
}

template <typename T>
void Prox<T>::get_separable_structure(
    vector<std::tuple<size_t, size_t, size_t> >& sep)
{
  sep.push_back( std::tuple<size_t, size_t, size_t> (index_, size_, 1) );
}


// Explicit template instantiation
template class Prox<float>;
template class Prox<double>;

} // namespace prost

