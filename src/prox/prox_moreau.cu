#include "prox_moreau.hpp"

#include "config.hpp"

template<typename T>
__global__
void MoreauPrescale(T *d_scaled_arg,
                    T *d_arg,
                    T *d_tau,
                    T tau,
                    size_t count,
                    bool invert_tau)
{ 
  int tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    if(invert_tau)
      d_scaled_arg[tx] = d_arg[tx] * (tau * d_tau[tx]);
    else
      d_scaled_arg[tx] = d_arg[tx] / (tau * d_tau[tx]);
  }
}

template<typename T>
__global__
void MoreauPostscale(T *d_result,
                     T *d_arg,
                     T *d_tau,
                     T tau,
                     size_t count,
                     bool invert_tau)
{ 
  int tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    if(invert_tau)
      d_result[tx] = d_arg[tx] - d_result[tx] / (tau * d_tau[tx]);
    else
      d_result[tx] = d_arg[tx] - tau * d_tau[tx] * d_result[tx];
  }
}

template<typename T>
ProxMoreau<T>::ProxMoreau(Prox *conjugate)
    : Prox(*conjugate), conjugate_(conjugate) {
}

template<typename T>
ProxMoreau<T>::~ProxMoreau() {
  Release();
}

template<typename T>
bool ProxMoreau<T>::Init() {
  cudaMalloc((void **)&d_scaled_arg_, sizeof(T) * (count_ * dim_));
  
  return cudaGetLastError() == CUDA_SUCCESS;
}

template<typename T>
void ProxMoreau<T>::Release() {
  cudaFree(d_scaled_arg_);
}

template<typename T>
void ProxMoreau<T>::EvalLocal(T *d_arg,
                              T *d_res,
                              T *d_tau,
                              T tau,
                              bool invert_tau)
{
  size_t total_count = count_ * dim_;
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((total_count + block.x - 1) / block.x, 1, 1);

  // scale argument
  MoreauPrescale<<<grid, block>>>(d_scaled_arg,
                                  d_arg,
                                  d_tau,
                                  tau,
                                  total_count,
                                  invert_tau);

  // compute prox with scaled argument
  conjugate_->EvalLocal(d_scaled_arg, d_result, d_tau, tau, !invert_tau);

  // combine back to get result of conjugate prox
  MoreauPostscale<<<grid, block>>>(d_result,
                                   d_arg,
                                   d_tau,
                                   tau,
                                   total_count,
                                   invert_tau);
}

template<typename T>
size_t ProxMoreau<T>::gpu_mem_amount() {
  return count_ * dim_ * sizeof(T);
}


// Explicit template instantiation
template class ProxMoreau<float>;
template class ProxMoreau<double>;
