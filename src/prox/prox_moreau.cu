#include "prox/prox_moreau.hpp"

#include <iostream>

template<typename T>
struct MoreauPrescale {
    const bool invert_tau_;
    const T tau_scal_;

    MoreauPrescale(bool invert_tau, T tau_scal) : invert_tau_(invert_tau), tau_scal_(tau_scal) {}

    __host__ __device__ T operator()(const T& arg, const T& tau_diag) const { 
        if(invert_tau_)
          return arg * (tau_scal_ * tau_diag);
        
        return arg / (tau_scal_ * tau_diag);
    }
};

template<typename T>
struct MoreauPostscale {
  const bool invert_tau_;
  const T tau_scal_;

  MoreauPostscale(bool invert_tau, T tau_scal) : invert_tau_(invert_tau), tau_scal_(tau_scal) {}

  template<typename Tuple>
  __host__ __device__ void operator()(Tuple t) const { 
    if(invert_tau_)
      thrust::get<2>(t) = thrust::get<0>(t) - thrust::get<2>(t) / (tau_scal_ * thrust::get<1>(t));
    else
      thrust::get<2>(t) = thrust::get<0>(t) - tau_scal_ * thrust::get<1>(t) * thrust::get<2>(t);
    }
};

template<typename T>
ProxMoreau<T>::ProxMoreau(std::unique_ptr<Prox<T>> conjugate)
    : Prox<T>(*conjugate), conjugate_(std::move(conjugate)) {} {
}

template<typename T>
ProxMoreau<T>::~ProxMoreau() {
  Release();
}

template<typename T>
void ProxMoreau<T>::Init() 
{
  conjugate_->Init();
}

template<typename T>
void ProxMoreau<T>::Release() 
{
  conjugate_->Release();
}

template<typename T>
void ProxMoreau<T>::EvalLocal(
  const thrust::device_ptr<T>& result,
  const thrust::device_ptr<const T>& arg,
  const thrust::device_ptr<const T>& tau_diag,
  T tau_scal,
  bool invert_tau)
{
  // prescale argument
  thrust::transform(
    arg, 
    arg + this->size_, 
    tau_diag, 
    scaled_arg_.begin(), 
    MoreauPrescale<T>(invert_tau, tau_scal));

  // compute prox with scaled argument
  conjugate_->EvalLocal(result, &scaled_arg_[0], tau_diag, tau_scal, !invert_tau);

  // postscale argument
  // combine back to get result of conjugate prox
  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(arg, tau_diag, result)),
    thrust::make_zip_iterator(
      thrust::make_tuple(
        arg + this->size_, 
        tau_diag + this->size_, 
        result + this->size_)),
    MoreauPostscale<T>(invert_tau, tau_scal));
}

template<typename T>
size_t ProxMoreau<T>::gpu_mem_amount() const {
  return this->size_ * sizeof(T) + conjugate_->gpu_mem_amount();
}

// Explicit template instantiation
template class ProxMoreau<float>;
template class ProxMoreau<double>;
