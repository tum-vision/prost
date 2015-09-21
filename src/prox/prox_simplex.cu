#include "prox/prox_simplex.hpp"

#include "config.hpp"
#include "util/cuwrap.hpp"

template<typename T>
__device__ void shellsort(T *a, int N) {
  const int gaps[6] = { 132, 57, 23, 10, 4, 1 };
  int i, j, k, gap;

  for(k = 0; k < 6; k++) {
    gap = gaps[k];

    for(i = gap; i < N; i++) {
      const T temp = a[i];

      for(j = i; (j >= gap) && (a[j - gap] <= temp); j -= gap) 
        a[j] = a[j - gap];

      a[j] = temp;
    }
  }
}

template<typename T>
__global__
void ProxSimplexKernel(T *d_arg,
                       T *d_res,
                       T *d_tau,
                       T tau,
                       T *d_coeffs,
                       size_t count,
                       size_t dim,
                       bool interleaved,
                       bool invert_tau)
{
  size_t tx, sx, i, index;
  extern __shared__ char sh_mem[];
  T *sh_arg = reinterpret_cast<T *>(sh_mem);

  tx = threadIdx.x + blockDim.x * blockIdx.x;
  
  if(tx >= count)
    return;
  
  // 1) read dim-dimensional vector into shared memory
  for(i = 0; i < dim; i++) {
    sx = threadIdx.x * dim + i;

    index = interleaved ? (tx * dim + i) : (tx + count * i);

    // handle inner product by completing the squaring and
    // pulling it into the squared term of the prox. while
    // scaling it correctly, taking are of the step size.
    T arg = d_arg[index];
    if(d_coeffs != NULL) {
      T tau_scaled = tau * d_tau[index];
      
      if(invert_tau)
        tau_scaled = 1. / tau_scaled;
      
      arg -= tau_scaled * d_coeffs[index];
    }
    
    sh_arg[sx] = arg;
  }
  __syncthreads();
  
  // 2) sort inside shared memory
  shellsort<T>(&sh_arg[threadIdx.x * dim], dim);

  bool bget = false;
  T tmpsum = 0;
  T tmax;
  for(int ii=1;ii<=dim-1;ii++) {
    tmpsum += sh_arg[threadIdx.x * dim + (ii - 1)];
    tmax = (tmpsum - 1.) / (T)ii;
    if(tmax >= sh_arg[threadIdx.x * dim + ii]){
      bget=true;
      break;
    }
  }

  if(!bget)
    tmax = (tmpsum + sh_arg[threadIdx.x*dim + (dim - 1)] - 1.0) / (T)dim;

  // 3) return result
  for(i = 0; i < dim; i++) {
    index = interleaved ? (tx * dim + i) : (tx + count * i);

    // handle inner product by completing the squaring and
    // pulling it into the squared term of the prox. while
    // scaling it correctly, taking are of the step size.
    T arg = d_arg[index];
    if(d_coeffs != NULL) {
      T tau_scaled = tau * d_tau[index];
      
      if(invert_tau)
        tau_scaled = 1. / tau_scaled;
      
      arg -= tau_scaled * d_coeffs[index];
    }

    d_res[index] = cuwrap::max<T>(arg - tmax, 0);
  }
}   

template<typename T>
ProxSimplex<T>::ProxSimplex(
    size_t index,
    size_t count,
    size_t dim,
    bool interleaved,
    const std::vector<T>& coeffs)

    : Prox<T>(index, count, dim, interleaved, false), coeffs_(coeffs)
{
}

template<typename T>
ProxSimplex<T>::~ProxSimplex() {
  Release();
}

template<typename T>
bool ProxSimplex<T>::Init() {
  if(coeffs_.empty())
    d_coeffs_ = NULL;
  else {
    cudaMalloc((void **)&d_coeffs_, sizeof(T) * this->count_);
    if(cudaGetLastError() != cudaSuccess) // out of memory
      return false;
    
    cudaMemcpy(d_coeffs_, &coeffs_[0], sizeof(T) * this->count_,
               cudaMemcpyHostToDevice);
  }

  // TODO: return false if not enough shared mem

  return true;
}

template<typename T>
void ProxSimplex<T>::Release() {
  cudaFree(d_coeffs_);
}

template<typename T>
void ProxSimplex<T>::EvalLocal(T *d_arg,
                               T *d_res,
                               T *d_tau,
                               T tau,
                               bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  size_t shmem_bytes = this->dim_ * block.x * sizeof(T);

  ProxSimplexKernel<T>
      <<<grid, block, shmem_bytes>>>(
          d_arg,
          d_res,
          d_tau,
          tau,
          d_coeffs_,
          this->count_,
          this->dim_,
          this->interleaved_,
          invert_tau);
}

template<typename T>
size_t ProxSimplex<T>::gpu_mem_amount() {
  if(d_coeffs_ != 0)
    return this->count_ * sizeof(T);
  else
    return 0;
}

// Explicit template instantiation
template class ProxSimplex<float>;
template class ProxSimplex<double>;
