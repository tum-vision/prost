#include "prox_simplex.hpp"
#include "util/cuwrap.hpp"

// TODO: move to util? best way to sort per thread in CUDA?
template<typename real>
__device__ void shellsort(real *a, int N) {
  const int gaps[6] = { 132, 57, 23, 10, 4, 1 };
  int i, j, k, gap;

  for(k = 0; k < 6; k++) {
    gap = gaps[k];

    for(i = gap; i < N; i++) {
      real temp = a[i];

      for(j = i; (j >= gap) && (a[j - gap] <= temp); j -= gap) 
        a[j] = a[j - gap];

      a[j] = temp;
    }
  }
}

// TODO: move to util?
template<typename real>
__device__ void bubblesort(real *a, int N) {

  for(int i = 0; i < N; i++) 
    for(int j = i + 1; j < N; j++) {
      if(a[i] <= a[j]) {
        real temp = a[i];
        a[i] = a[j];
        a[j] = temp;
      }
    }
}

template<typename real>
__global__
void ProxSimplexKernel(
    const real *d_arg,
    real *d_result,
    real tau,
    const real *d_tau,
    const real *d_coeffs,
    int index,
    int count,
    int dim,
    bool interleaved,
    bool invert_step) {

  int th_idx;
  int sh_idx;
  int global_idx;
  int i;
  extern __shared__ real sh_y[];

  th_idx = threadIdx.x + blockDim.x * blockIdx.x;
  
  if(th_idx >= count)
    return;
  
  // 1) read dim-dimensional vector into shared memory
  for(i = 0; i < dim; i++) {
    sh_idx = threadIdx.x * dim + i;
    
    if(interleaved)
      global_idx = index + th_idx * dim + i;
    else
      global_idx = index + th_idx + count * i;

    // handle inner product by completing the squaring and
    // pulling it into the squared term of the prox. while
    // scaling it correctly, taking are of the step size.
    real arg = d_arg[global_idx];
    if(d_coeffs != NULL)
    {
      real new_tau = tau;
      
      if(d_tau != NULL)
        new_tau *= d_tau[global_idx];

      if(invert_step)
        new_tau = 1.0 / new_tau;
      
      arg -= new_tau * d_coeffs[global_idx];
    }
    
    sh_y[sh_idx] = arg;
  }
  __syncthreads();
  
  // 2) sort inside shared memory
  shellsort<real>(&sh_y[threadIdx.x * dim], dim);
  //bubblesort<real>(&sh_y[threadIdx.x * dim], dim);

  bool bget = false;
  real tmpsum = 0;
  real tmax;
  for(int ii=1;ii<=dim-1;ii++) {
    tmpsum += sh_y[threadIdx.x * dim + (ii - 1)];
    tmax = (tmpsum - 1.0) / (real)ii;
    if(tmax >= sh_y[threadIdx.x * dim + ii]){
      bget=true;
      break;
    }
  }

  if(!bget)
    tmax = (tmpsum + sh_y[threadIdx.x*dim + (dim - 1)] - 1.0) / (real)dim;

  // 5) return result
  for(i = 0; i < dim; i++) {
    if(interleaved)
      global_idx = index + th_idx * dim + i;
    else
      global_idx = index + th_idx + count * i;

    // handle inner product by completing the squaring and
    // pulling it into the squared term of the prox. while
    // scaling it correctly, taking are of the step size.
    real arg = d_arg[global_idx];
    if(d_coeffs != NULL)
    {
      real new_tau = tau;
      
      if(d_tau != NULL)
        new_tau *= d_tau[global_idx];

      if(invert_step)
        new_tau = 1.0 / new_tau;
      
      arg -= new_tau * d_coeffs[global_idx];
    }

    d_result[global_idx] = cuwrap::max<real>(arg - tmax, 0);
  }
}   

ProxSimplex::ProxSimplex(
    int index,
    int count,
    int dim,
    bool interleaved,
    const std::vector<real>& coeffs)

    : Prox(index, count, dim, interleaved, false), coeffs_(coeffs)
{
  if(coeffs_.empty())
    d_coeffs_ = NULL;
  else {
    cudaMalloc((void **)&d_coeffs_, sizeof(real) * count_);
  
    // TODO: this assumes linear storage -- always holds?
    cudaMemcpy(d_coeffs_, &coeffs_[0], sizeof(real) * count_, cudaMemcpyHostToDevice);
  }
}

ProxSimplex::~ProxSimplex() {
  cudaFree(d_coeffs_);
}

void ProxSimplex::Evaluate(
    real *d_arg,
    real *d_result,
    real tau,
    real *d_tau,
    bool invert_step) {

  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((count_ + block.x - 1) / block.x, 1, 1);

  size_t shmem_bytes = dim_ * block.x * sizeof(real);

  ProxSimplexKernel<real>
      <<<grid, block, shmem_bytes>>>(
          d_arg,
          d_result,
          tau,
          d_tau,
          d_coeffs_,
          index_,
          count_,
          dim_,
          interleaved_,
          invert_step);
  
  cudaDeviceSynchronize();
}

int ProxSimplex::gpu_mem_amount() {
  if(d_coeffs_ != 0)
    return count_ * sizeof(real);
  else
    return 0;
}
