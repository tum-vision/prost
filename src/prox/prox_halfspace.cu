#include "prox/prox_halfspace.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include "config.hpp"
#include <iostream>

using namespace std;

template<typename T>
__global__
void ProxHalfspaceKernel(T *d_arg,
                            T *d_res,
                            T *d_ptr_a,
                            size_t count,
                            size_t dim,
                            bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    T dot = 0, sq_norm = 0;
    size_t index;
    for(size_t i = 0; i < dim; i++) {
      index = interleaved ? (tx * dim + i) : (tx + count * i);
      T a = d_ptr_a[index];
      dot += d_arg[index]*a;
      sq_norm += a*a;
    }

    if(dot > 0) {
      const T s = dot / sq_norm;
      for(size_t i = 0; i < dim; i++) {
        index = interleaved ? (tx * dim + i) : (tx + count * i);
        d_res[index] = d_arg[index] - s * d_ptr_a[index];
      }
    } else {
      for(size_t i = 0; i < dim; i++) {
        index = interleaved ? (tx * dim + i) : (tx + count * i);
        d_res[index] = d_arg[index];
      } 
    }
  }
}

template<typename T>
ProxHalfspace<T>::ProxHalfspace(size_t index,
                                      size_t count,
                                      size_t dim,
                                      bool interleaved,
                                      std::vector<T>& a)
    
    : Prox<T>(index, count, dim, interleaved, false), a_(a)
{
}

template<typename T>
ProxHalfspace<T>::~ProxHalfspace() {
  Release();
}

template<typename T>
bool ProxHalfspace<T>::Init() {
  if(a_.size() != this->count_*this->dim_)
    return false;  

  T *d_ptr_a = NULL;

  // copy b
  size_t size = a_.size() * sizeof(T);

  // copy a
  cudaMalloc((void **)&d_ptr_a, size);
  cudaError err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  cudaMemcpy(d_ptr_a, &a_[0], size, cudaMemcpyHostToDevice);
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cout << cudaGetErrorString(err)<< std::endl;            
    return false;
  }
  d_ptr_a_ = d_ptr_a;

  return true;
}

template<typename T>
void ProxHalfspace<T>::Release() {
  cudaFree(d_ptr_a_);
}

template<typename T>
void ProxHalfspace<T>::EvalLocal(T *d_arg,
                                    T *d_res,
                                    T *d_tau,
                                    T tau,
                                    bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  ProxHalfspaceKernel<T>
      <<<grid, block>>>(
          d_arg,
          d_res,
          d_ptr_a_,
          this->count_,
          this->dim_,
          this->interleaved_);
}

// Explicit template instantiation
template class ProxHalfspace<float>;
template class ProxHalfspace<double>;
