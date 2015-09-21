#include "prox/prox_epi_conjquadr.hpp"

#include <cassert>
#include <cuda_runtime.h>
#include "config.hpp"

template<typename T>
__global__
void ProxEpiConjQuadrKernel(T *d_arg,
                            T *d_res,
                            EpiConjQuadrCoeffsDevice<T> coeffs,
                            size_t count,
                            bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    T result[2];

    // get v = (x0, y0) and a,b,c,alpha,beta
    T a, b, c, alpha, beta;
    T v[2];

    a = coeffs.d_ptr[0][tx];
    b = coeffs.d_ptr[1][tx];
    c = coeffs.d_ptr[2][tx];
    alpha = coeffs.d_ptr[3][tx];
    beta = coeffs.d_ptr[4][tx];

    if(interleaved) {
      v[0] = d_arg[tx * 2 + 0];
      v[1] = d_arg[tx * 2 + 1];
    }
    else {
      v[0] = d_arg[tx + count * 0];
      v[1] = d_arg[tx + count * 1];
    }
    
    // check which case applies (0 = A, 1 = B, 2 = C)
    const T p_A[2] = { 2 * a * alpha + b, a * alpha * alpha - c };
    const T p_B[2] = { 2 * a * beta + b, a * beta * beta - c };
    T n_A[2] = { 1, alpha };
    T n_B[2] = { -1, -beta }; 
    
    int proj_case;
    if(PointInHalfspace(v, p_A, n_A, 2))
      proj_case = 0;
    else if(PointInHalfspace(v, p_B, n_B, 2))
      proj_case = 2;
    else
      proj_case = 1;
    
    // perform projection
    switch(proj_case) {
      case 0: { // case A
        n_A[0] = -alpha;
        n_A[1] = 1.;
        const T t = -a * alpha * alpha - b * alpha - c;
          
        ProjectHalfspace<T>(v,
                            n_A,
                            t,
                            result,
                            2);
      } break;

      case 1: { // case B
        if(a > 0) 
          ProjectParabolaGeneral<T>(v[0],
                                    v[1],
                                    1. / (4. * a),
                                    -b / (2. * a),
                                    b * b / (4. * a) - c,
                                    result[0],
                                    result[1]);
        else {
          // if a <= 0 the parabola disappears and we're in the normal cone.
          result[0] = a * (alpha + beta) + b;
          result[1] = alpha * beta * a - c;
        }
          
      } break;

      case 2: { // case C
        n_B[0] = -beta;
        n_B[1] = 1.;
        const T t = -a * beta * beta - b * beta - c;

        ProjectHalfspace<T>(v,
                            n_B,
                            t,
                            result,
                            2);
      } break;
    }      
    
    // write result
    if(interleaved) {
      d_res[tx * 2 + 0] = result[0];
      d_res[tx * 2 + 1] = result[1];
    }
    else {
      d_res[tx + count * 0] = result[0];
      d_res[tx + count * 1] = result[1];
    }
  }
}

template<typename T>
ProxEpiConjQuadr<T>::ProxEpiConjQuadr(size_t index,
                                      size_t count,
                                      bool interleaved,
                                      const EpiConjQuadrCoeffs<T>& coeffs)
    
    : Prox<T>(index, count, 2, interleaved, false), coeffs_(coeffs)
{
}

template<typename T>
ProxEpiConjQuadr<T>::~ProxEpiConjQuadr() {
  Release();
}

template<typename T>
bool ProxEpiConjQuadr<T>::Init() {
  std::vector<T>* coeff_array[PROX_EPI_CONJQUADR_NUM_COEFFS] = {
    &coeffs_.a,
    &coeffs_.b,
    &coeffs_.c,
    &coeffs_.alpha,
    &coeffs_.beta };
  
  for(size_t i = 0; i < PROX_EPI_CONJQUADR_NUM_COEFFS; i++) {
    const std::vector<T>& cur_elem = *(coeff_array[i]);

    if(cur_elem.empty())
      return false;
    
    T *gpu_ptr = NULL;
    cudaMalloc((void **)&gpu_ptr, this->count_ * sizeof(T));
    if(cudaGetLastError() != cudaSuccess)
      return false;
    
    cudaMemcpy(gpu_ptr, &cur_elem[0], sizeof(T) * this->count_, cudaMemcpyHostToDevice);
    coeffs_dev_.d_ptr[i] = gpu_ptr;
  }
  return true;
}

template<typename T>
void ProxEpiConjQuadr<T>::Release() {
  for(size_t i = 0; i < PROX_EPI_CONJQUADR_NUM_COEFFS; i++) {
    cudaFree(coeffs_dev_.d_ptr[i]);
  }
}

template<typename T>
void ProxEpiConjQuadr<T>::EvalLocal(T *d_arg,
                                    T *d_res,
                                    T *d_tau,
                                    T tau,
                                    bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  ProxEpiConjQuadrKernel<T>
      <<<grid, block>>>(
          d_arg,
          d_res,
          coeffs_dev_,
          this->count_,
          this->interleaved_);
}

// Explicit template instantiation
template class ProxEpiConjQuadr<float>;
template class ProxEpiConjQuadr<double>;
