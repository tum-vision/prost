#include "prost/prox/prox_ind_epi_polyhedral_1d.hpp"

namespace prost {

template<typename T>
ProxIndEpiPolyhedral1D<T>::ProxIndEpiPolyhedral1D(
  size_t index,
  size_t count,
  bool interleaved,
  const vector<T>& pt_x, 
  const vector<T>& pt_y,
  const vector<T>& alpha,
  const vector<T>& beta,
  const vector<size_t>& count,
  const vector<size_t>& index)
{
}

template<typename T>
void ProxIndEpiPolyhedral1D<T>::Initialize()
{
}

template<typename T>
size_t ProxIndEpiPolyhedral1D<T>::gpu_mem_amount() const
{
  return -1; // TODO
}

template<typename T>
void ProxIndEpiPolyhedral1D<T>::EvalLocal(
  const typename device_vector<T>::iterator& result_beg,
  const typename device_vector<T>::iterator& result_end,
  const typename device_vector<T>::const_iterator& arg_beg,
  const typename device_vector<T>::const_iterator& arg_end,
  const typename device_vector<T>::const_iterator& tau_beg,
  const typename device_vector<T>::const_iterator& tau_end,
  T tau,
  bool invert_tau)
{
}

template class ProxIndEpiPolyhedral1D<float>;
template class ProxIndEpiPolyhedral1D<double>;

} // namespace prost
