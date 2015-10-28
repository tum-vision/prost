#include "prox/prox_1d.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <cuda_runtime.h>

#include "config.hpp"

template<class Prox1DFunc, typename T>
__global__
void Prox1DKernel(
    T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    const Prox1DFunc prox,
    Prox1DCoeffsDevice<T> coeffs,
    size_t count,
    bool invert_tau)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) { 

    // read value for vector coefficients
    for(size_t i = 0; i < PROX_1D_NUM_COEFFS; i++) {
      if(coeffs.d_ptr[i] != NULL)
        coeffs.val[i] = coeffs.d_ptr[i][tx];
    }

    if(coeffs.val[2] == 0) // c == 0 -> prox_zero -> return argument
      d_res[tx] = d_arg[tx];
    else
    {
      // compute step-size
      tau = invert_tau ? (1. / (tau * d_tau[tx])) : (tau * d_tau[tx]);

      // compute scaled prox argument and step 
      const T arg = ((coeffs.val[0] * (d_arg[tx] - coeffs.val[3] * tau)) /
        (1. + tau * coeffs.val[4])) - coeffs.val[1];
    
      const T step = (coeffs.val[2] * coeffs.val[0] * coeffs.val[0] * tau) /
        (1. + tau * coeffs.val[4]);

      // compute scaled prox and store result
      d_res[tx] = 
        (prox.Eval(arg, step, coeffs.val[5], coeffs.val[6]) + coeffs.val[1])
        / coeffs.val[0];
    }
  }
}

template<typename T>
Prox1D<T>::Prox1D(size_t index,
                  size_t count,
                  const Prox1DCoefficients<T>& coeffs,
                  const Prox1DFunction& func)
    : Prox<T>(index, count, 1, false, true), coeffs_(coeffs), func_(func)
{
}

template<typename T>
Prox1D<T>::~Prox1D() {
  Release();
}

template<typename T>
bool Prox1D<T>::Init() {
  
  std::vector<T>* coeff_array[PROX_1D_NUM_COEFFS] =
      { &coeffs_.a,
        &coeffs_.b,
        &coeffs_.c,
        &coeffs_.d,
        &coeffs_.e,
        &coeffs_.alpha,
        &coeffs_.beta };

  for(size_t i = 0; i < PROX_1D_NUM_COEFFS; i++) {
    const std::vector<T>& cur_elem = *(coeff_array[i]);

    if(cur_elem.empty())
      return false;

    T *gpu_data = NULL;
    
    // if there is more than 1 coeff store it in a global mem.
    if(cur_elem.size() > 1) {

      // check if dimension is correct
      if(cur_elem.size() != this->count_)
        return false;

      cudaMalloc((void **)&gpu_data, sizeof(T) * this->count_);
      if(cudaGetLastError() != cudaSuccess) // out of memory
        return false;
      
      cudaMemcpy(gpu_data, &cur_elem[0], sizeof(T) * this->count_, cudaMemcpyHostToDevice);
    }
    else 
      coeffs_dev_.val[i] = cur_elem[0];

    // for single coeffs, NULL is pushed back.
    coeffs_dev_.d_ptr[i] = gpu_data;
  }

  return true;
}

template<typename T>
void Prox1D<T>::Release() {
  for(int i = 0; i < PROX_1D_NUM_COEFFS; i++)
    if(coeffs_dev_.d_ptr[i])
      cudaFree(coeffs_dev_.d_ptr[i]);
}

template<typename T>
void Prox1D<T>::EvalLocal(T *d_arg,
                          T *d_res,
                          T *d_tau,
                          T tau,
                          bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  // makes things more readable
  #define CALL_PROX_1D_KERNEL(Func) \
    Prox1DKernel<Func<T>, T> \
        <<<grid, block>>>( \
             d_arg, \
             d_res, \
             d_tau, \
             tau, \
             Func<T>(), \
             coeffs_dev_, \
             this->count_, invert_tau)
  
  switch(func_) {    
    case kZero: 
      CALL_PROX_1D_KERNEL(Prox1DZero);
      break;
      
    case kAbs: 
      CALL_PROX_1D_KERNEL(Prox1DAbs);
      break;

    case kSquare: 
      CALL_PROX_1D_KERNEL(Prox1DSquare);
      break;

    case kMaxPos0:
      CALL_PROX_1D_KERNEL(Prox1DMaxPos0);
      break;

    case kIndLeq0: 
      CALL_PROX_1D_KERNEL(Prox1DIndLeq0);
      break;

    case kIndGeq0: 
      CALL_PROX_1D_KERNEL(Prox1DIndGeq0);
      break;

    case kIndEq0: 
      CALL_PROX_1D_KERNEL(Prox1DIndEq0);
      break;

    case kIndBox01: 
      CALL_PROX_1D_KERNEL(Prox1DIndBox01);
      break;

    default:
      break;
  }
}

template<typename T>
size_t Prox1D<T>::gpu_mem_amount() {
  size_t total_mem = 0;
  for(size_t i = 0; i < PROX_1D_NUM_COEFFS; i++) {
    if(coeffs_dev_.d_ptr[i] != NULL)
      total_mem += this->count_ * sizeof(T);
  }

  return total_mem;
}

// Explicit template instantiation
template class Prox1D<float>;
template class Prox1D<double>;
