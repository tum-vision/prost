#include "prox/prox_moreau.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include "config.hpp"

using namespace prox;

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
            get<2>(t) = get<0>(t) - get<2>(t) / (tau_scal_ * get<1>(t));
        else
            get<2>(t) = get<0>(t) - tau_scal_ * get<1>(t) * get<2>(t);
    }
};

template<typename T>
ProxMoreau<T>::ProxMoreau(std::unique_ptr<Prox<T>> conjugate)
    : Prox<T>(*conjugate), conjugate_(std::move(conjugate)) {} {
}

template<typename T>
void ProxMoreau<T>::Init() {
  try {
    d_scaled_arg_.resize(this->size_);
  } catch(std::bad_alloc &e) {
    throw PDSolverException();
  } catch(thrust::system_error &e) {
    throw PDSolverException();
  }

  conjugate_->Init();
}

template<typename T>
void ProxMoreau<T>::EvalLocal(typename thrust::device_vector<T>::iterator d_arg_begin,
                         typename thrust::device_vector<T>::iterator d_arg_end,
                         typename thrust::device_vector<T>::iterator d_res_begin,
                         typename thrust::device_vector<T>::iterator d_res_end,
                         typename thrust::device_vector<T>::iterator d_tau_begin,
                         typename thrust::device_vector<T>::iterator d_tau_end,
                         T tau,
                         bool invert_tau)
{

  // prescale argument
  thrust::transform(d_arg_begin, d_arg_end, d_tau_begin, d_scaled_arg_.begin(), MoreauPrescale<T>(invert_tau, tau));

  // compute prox with scaled argument
  conjugate_->EvalLocal(d_scaled_arg_.begin(), d_scaled_arg_.end(), d_res_begin, d_res_end, d_tau_begin, d_tau_end, tau, !invert_tau);

  // postscale argument
  // combine back to get result of conjugate prox
  thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(d_arg_begin, d_tau_begin, d_res_begin)),
           thrust::make_zip_iterator(thrust::make_tuple(d_arg_end, d_tau_end, d_res_end)),
           MoreauPostscale<T>(invert_tau, tau));
}

template<typename T>
size_t ProxMoreau<T>::gpu_mem_amount() {
  return this->size_ * sizeof(T) + conjugate_->gpu_mem_amount();
}

// Explicit template instantiation
template class ProxMoreau<float>;
template class ProxMoreau<double>;
