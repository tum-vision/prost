#include <iostream>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include "prost/prox/prox_moreau.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
struct MoreauPrescale 
{
  const bool invert_tau_;
  const T tau_scal_;

  MoreauPrescale(bool invert_tau, T tau_scal) : invert_tau_(invert_tau), tau_scal_(tau_scal) { }

  __host__ __device__ T operator()(const T& arg, const T& tau_diag) const 
  { 
    if(invert_tau_)
      return arg * (tau_scal_ * tau_diag);
        
    return arg / (tau_scal_ * tau_diag);
  }
};

template<typename T>
struct MoreauPostscale {
  const bool invert_tau_;
  const T tau_scal_;

  MoreauPostscale(bool invert_tau, T tau_scal) : invert_tau_(invert_tau), tau_scal_(tau_scal) { }

  template<typename Tuple>
  __host__ __device__ void operator()(Tuple t) const 
  { 
    if(invert_tau_)
      thrust::get<2>(t) = thrust::get<0>(t) - thrust::get<2>(t) / (tau_scal_ * thrust::get<1>(t));
    else
      thrust::get<2>(t) = thrust::get<0>(t) - tau_scal_ * thrust::get<1>(t) * thrust::get<2>(t);
  }
};

template<typename T>
ProxMoreau<T>::ProxMoreau(std::shared_ptr<Prox<T> > conjugate)
  : Prox<T>(*conjugate), conjugate_(conjugate) 
{
}

template<typename T>
ProxMoreau<T>::~ProxMoreau() 
{
}

template<typename T>
void ProxMoreau<T>::Initialize() 
{
  try 
  {
    scaled_arg_.resize(this->size_);
  } 
  catch(const std::bad_alloc &e)
  {
	std::stringstream ss;
	ss << "Out of memory: " << e.what();

    throw Exception(ss.str());
  }

  conjugate_->Initialize();
}

template<typename T>
void ProxMoreau<T>::Release() 
{
  conjugate_->Release();
}

template<typename T>
void ProxMoreau<T>::EvalLocal(
  const typename thrust::device_vector<T>::iterator& result_beg,
  const typename thrust::device_vector<T>::iterator& result_end,
  const typename thrust::device_vector<T>::const_iterator& arg_beg,
  const typename thrust::device_vector<T>::const_iterator& arg_end,
  const typename thrust::device_vector<T>::const_iterator& tau_beg,
  const typename thrust::device_vector<T>::const_iterator& tau_end,
  T tau,
  bool invert_tau)
{
  // prescale argument
  thrust::transform(
    arg_beg, 
    arg_end,
    tau_beg, 
    scaled_arg_.begin(), 
    MoreauPrescale<T>(invert_tau, tau));

  // compute prox with scaled argument
  conjugate_->EvalLocal(
    result_beg, 
    result_end,
    scaled_arg_.begin(),
    scaled_arg_.end(),
    tau_beg,
    tau_end,
    tau, 
    !invert_tau);

  // postscale argument
  // combine back to get result of conjugate prox
  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(arg_beg, tau_beg, result_beg)),
    thrust::make_zip_iterator(thrust::make_tuple(arg_end, tau_end, result_end)),
    MoreauPostscale<T>(invert_tau, tau));
}

template<typename T>
size_t ProxMoreau<T>::gpu_mem_amount() const 
{
  return this->size_ * sizeof(T) + conjugate_->gpu_mem_amount();
}

template<typename T>
void ProxMoreau<T>::get_separable_structure(
  vector<std::tuple<size_t, size_t, size_t> >& sep)
{
  conjugate_->get_separable_structure(sep);
}


// Explicit template instantiation
template class ProxMoreau<float>;
template class ProxMoreau<double>;

} // namespace prost
