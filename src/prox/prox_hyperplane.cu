#include "prox/prox_hyperplane.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include "config.hpp"
#include <iostream>

using namespace std;

template<typename T>
__global__
void ProxHyperplaneKernel(T *d_arg,
                            T *d_res,
                            T *d_ptr_b,
                            size_t index,
                            size_t count,
                            size_t dim)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;


  if(tx < count) {
    size_t d = dim-1;
    T r = d_arg[count*d + tx];

    T dot = 0, sq_norm = 0;
    for(size_t i = 0; i < d; i++) {
      T b = d_ptr_b[count*i + tx];
      dot += d_arg[count*i + tx]*b;
      sq_norm += b*b;
    }

    if(dot > r) {
      const T lambda = (dot - r) / (1 + sq_norm);
      for(size_t i = 0; i < d; i++) 
        d_res[i*count + tx] = d_arg[i*count + tx] - lambda * d_ptr_b[i*count + tx];

      d_res[d*count + tx] = r + lambda;
    } else {
      for(size_t i = 0; i < d; i++) 
        d_res[i*count + tx] = d_arg[i*count + tx];

      d_res[d*count + tx] = r;     
    }
  }
}

template<typename T>
ProxHyperplane<T>::ProxHyperplane(size_t index,
                                      size_t count,
                                      size_t dim,
                                      std::vector<T>& b)
    
    : Prox<T>(index, count, dim, true, false), b_(b)
{
}

template<typename T>
ProxHyperplane<T>::~ProxHyperplane() {
  Release();
}

template<typename T>
bool ProxHyperplane<T>::Init() {
  if(b_.size() != this->count_*(this->dim_-1))
    return false;  

  T *d_ptr_b = NULL;

  // copy b
  size_t size = b_.size() * sizeof(T);

  // copy b
  cudaMalloc((void **)&d_ptr_b, size);
  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  cudaMemcpy(d_ptr_b, &b_[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  d_ptr_b_ = d_ptr_b;

  return true;
}

template<typename T>
void ProxHyperplane<T>::Release() {
  cudaFree(d_ptr_b_);
}

template<typename T>
void ProxHyperplane<T>::EvalLocal(T *d_arg,
                                    T *d_res,
                                    T *d_tau,
                                    T tau,
                                    bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  ProxHyperplaneKernel<T>
      <<<grid, block>>>(
          d_arg,
          d_res,
          d_ptr_b_,
          this->index_,
          this->count_,
          this->dim_);
}

// Explicit template instantiation
template class ProxHyperplane<float>;
template class ProxHyperplane<double>;
