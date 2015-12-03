#include "prox/prox_epi_huber.hpp"

#include <cassert>
#include <iostream>
#include <cuda_runtime.h>
#include "config.hpp"

using namespace std;

#define MAX_DIM  2

template<typename T>
__global__ void
ProxEpiHuberKernel(
  T *d_arg,
  T *d_res,
  T *d_g_,
  size_t count,
  size_t dim,
  T alpha,
  T g)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) { // check if active
    // load input
    T phi_x[MAX_DIM];

    T norm_phi_x = 0;
    for(size_t i = 0; i < dim; i++) {
      phi_x[i] = d_arg[tx + i * count];
      norm_phi_x += phi_x[i] * phi_x[i];
    }
    norm_phi_x = sqrt(norm_phi_x);

    T phi_t = d_arg[tx + dim * count];

    // storage for result
    T res_x[MAX_DIM];
    T res_y;

    if( (alpha / 2.) <= phi_t ) // case 1 in SIIMS'10
    {
      for(size_t i = 0; i < dim; i++) {
        res_x[i] = phi_x[i] / max(static_cast<T>(1), norm_phi_x);
      }

      res_y = phi_t;
    }
    else if( phi_t < ((alpha / 2.) - (norm_phi_x - 1.) / alpha) ) // case 3 
    {
      // project on parabola 
      if(d_g_ != 0)
        ProjectParabolaShiftedNd<T>(phi_x, phi_t, alpha, d_g_[tx], res_x, res_y, dim);
      else
        ProjectParabolaShiftedNd<T>(phi_x, phi_t, alpha, g, res_x, res_y, dim);    
    }
    else // case 2
    {
      for(size_t i = 0; i < dim; i++) {
        res_x[i] = phi_x[i] / max(static_cast<T>(1), norm_phi_x);
      }

      res_y = alpha / 2.;
    }     

    // write out result
    for(size_t i = 0; i < dim; i++)
      d_res[tx + i * count] = res_x[i];
    
    d_res[tx + dim*count] = res_y;
  }
}

template<typename T>
ProxEpiHuber<T>::ProxEpiHuber(
  size_t index,
  size_t count,
  size_t dim,
  const std::vector<T>& g,
  T alpha)

  : Prox<T>(index, count, dim, false, false), g_(g), alpha_(alpha)
{
}

template<typename T>
ProxEpiHuber<T>::~ProxEpiHuber()
{
  Release();
}

template<typename T>
bool 
ProxEpiHuber<T>::Init()
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
void 
ProxEpiHuber<T>::Release()
{
  if(d_g_ != 0)
    cudaFree(d_g_);
}
  
template<typename T>
void 
ProxEpiHuber<T>::EvalLocal(T *d_arg, T *d_res, T *d_tau, T tau, bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  ProxEpiHuberKernel<T>
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
template class ProxEpiHuber<float>;
template class ProxEpiHuber<double>;
