#include "prox/prox_moreau.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include "config.hpp"

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
        if(invert_tau)
            get<2>(t) = get<0>(t) - get<2>(t) / (tau_scal_ * get<1>(t));
        else
            get<2>(t) = get<0>(t) - tau_scal_ * get<1>(t) * get<2>(t);
    }
};

template<typename T>
ProxMoreau<T>::ProxMoreau(unique_ptr<Prox<T>> conjugate)
    : Prox<T>(*conjugate), conjugate_(std::move(conjugate)) {} {
}

template<typename T>
ProxMoreau<T>::~ProxMoreau() {
  Release();
}

template<typename T>
bool ProxMoreau<T>::Init() {
  return conjugate_->Init();
}

template<typename T>
void ProxMoreau<T>::Release() {
}

template<typename T>
void ProxMoreau<T>::EvalLocal(device_vector<T> d_arg,
                              device_vector<T> d_res,
                              device_vector<T> d_tau,
                              T tau,
                              bool invert_tau)
{

  // prescale argument
  transform(d_arg.begin(), d_arg.end(), d_tau.begin(), d_scaled_arg_.begin(), MoreauPrescale(invert_tau, tau));

  // compute prox with scaled argument
  conjugate_->EvalLocal(d_scaled_arg_, d_res, d_tau, tau, !invert_tau);

  // postscale argument
  // combine back to get result of conjugate prox
  for_each(make_zip_iterator(make_tuple(d_arg.begin(), d_tau.begin(), d_res.begin())),
           make_zip_iterator(make_tuple(d_arg.end(), d_tau.end(), d_res.end())),
           MoreauPostscale(invert_tau, tau));
}

template<typename T>
size_t ProxMoreau<T>::gpu_mem_amount() {
  return this->size_ * sizeof(T);
}

// Explicit template instantiation
template class ProxMoreau<float>;
template class ProxMoreau<double>;
