#include "prox/prox_plus_linterm.hpp"

#include <iostream>
#include <cuda_runtime.h>
#include "config.hpp"

template<typename T>
__global__
void ShiftProxArg(
  T *d_arg,
  T *d_c,
  T *d_tau,
  T tau,
  size_t count,
  bool invert_tau) 
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    if(invert_tau)
      d_arg[tx] = d_arg[tx] - d_c[tx] / (tau * d_tau[tx]);
    else
      d_arg[tx] = d_arg[tx] - d_c[tx] * (tau * d_tau[tx]);

    //d_arg[tx] = d_arg[tx];
  }
} 

template<typename T>
ProxPlusLinterm<T>::ProxPlusLinterm(
  Prox<T> *prox,
  const std::vector<T>& c) 
  : Prox<T>(*prox), prox_(prox), c_(c), d_c_(0)
{
}

template<typename T>
ProxPlusLinterm<T>::~ProxPlusLinterm() {
  Release();
}

template<typename T>
bool 
ProxPlusLinterm<T>::Init() {
  bool success = prox_->Init();

  if(this->count_ * this->dim_ != c_.size() &&
    c_.size() != 1)
  {
    std::cout << "Invalid size of linterm.\n";
    return false;
  }

  cudaMalloc((void **)&d_c_, sizeof(T) * (this->count_ * this->dim_));
  cudaMemcpy(d_c_, &c_[0], sizeof(T) * (this->count_ * this->dim_), cudaMemcpyHostToDevice);

  return success && (cudaGetLastError() == cudaSuccess);
}

template<typename T>
void 
ProxPlusLinterm<T>::Release() {
  cudaFree(d_c_);
}

template<typename T>
size_t 
ProxPlusLinterm<T>::gpu_mem_amount() {
  return prox_->gpu_mem_amount();
}

template<typename T>
void 
ProxPlusLinterm<T>::EvalLocal(
    T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau)
{
  size_t total_count = this->count_ * this->dim_;
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((total_count + block.x - 1) / block.x, 1, 1);

  ShiftProxArg<T>
    <<<grid, block>>>(
      d_arg,
      d_c_,
      d_tau,
      tau,
      total_count,
      invert_tau);
  
  prox_->EvalLocal(d_arg, d_res, d_tau, tau, invert_tau);
}

// Explicit template instantiation
template class ProxPlusLinterm<float>;
template class ProxPlusLinterm<double>;
