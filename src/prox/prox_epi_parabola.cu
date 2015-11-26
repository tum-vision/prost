#include "prox/prox_epi_parabola.hpp"

#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include "config.hpp"

using namespace std;

#define MAX_DIM  2

template<typename T>
__global__ void
ProxEpiParabolaKernel(
  T *d_arg,
  T *d_res,
  T *d_g_,
  size_t count,
  size_t dim,
  T alpha,
  T g)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    T x0[MAX_DIM];

    for(size_t i = 0; i < dim; i++)
      x0[i] = d_arg[tx + i * count];

    T res_x[MAX_DIM];
    T res_y;

    if(d_g_ != 0)
      ProjectParabolaShiftedNd<T>(x0, d_arg[tx + dim * count], alpha, d_g_[tx], res_x, res_y, dim);
    else
      ProjectParabolaShiftedNd<T>(x0, d_arg[tx + dim * count], alpha, g, res_x, res_y, dim);    

    for(size_t i = 0; i < dim; i++)
      d_res[tx + i * count] = res_x[i];

    d_res[tx + dim*count] = res_y;
  }
}


template<typename T>
ProxEpiParabola<T>::ProxEpiParabola(
  size_t index,
  size_t count,
  size_t dim,
  const std::vector<T>& g,
  T alpha)
  : Prox<T>(index, count, dim, false, false), g_(g), alpha_(alpha)
{
}

template<typename T>
ProxEpiParabola<T>::~ProxEpiParabola() 
{
  Release();
}

template<typename T>
bool ProxEpiParabola<T>::Init() 
{
  if(this->count_ != g_.size() && g_.size() != 1)
  {
    std::cout << "Invalid number of coefficients (count=" << this->count_ << ", size=" << g_.size() << ")." << std::endl;
    return false;
  }

  if(this->dim_ - 1 > MAX_DIM)
  {
    std::cout << "Currently only dimension of size " << MAX_DIM << " supported." << std::endl;
  }

  if(g_.size() > 1)
  {
    cudaMalloc((void **)&d_g_, this->count_ * sizeof(T));
    if(cudaGetLastError() != cudaSuccess)
      return false;

    cudaMemcpy(d_g_, &g_[0], sizeof(T) * this->count_, cudaMemcpyHostToDevice);
  }
  else
  {
    d_g_ = 0;
  }

  return true;
}

template<typename T>
void ProxEpiParabola<T>::Release() 
{
  if(d_g_ != 0)
    cudaFree(d_g_);
}

template<typename T>
void ProxEpiParabola<T>::EvalLocal(
  T *d_arg,
  T *d_res,
  T *d_tau,
  T tau,
  bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  ProxEpiParabolaKernel<T>
      <<<grid, block>>>(
        d_arg,
        d_res,
        d_g_,
        this->count_,
        this->dim_ - 1,
        alpha_,
        g_[0]);
}

// Explicit template instantiation
template class ProxEpiParabola<float>;
template class ProxEpiParabola<double>;
