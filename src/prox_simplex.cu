#include "prox_simplex.hpp"
#include "util/cuwrap.hpp"

// TODO: move to util? best way to sort per thread in CUDA?
template<typename real>
__device__ void shellsort(real *data, int N) {
  const int gaps[6] = { 132, 57, 23, 10, 4, 1 };
  int i, j, k, gap;

  for(k = 0; k < 6; k++) {
    gap = gaps[k];

    for(i = gap; i < N; i++) {
      real temp = data[i];

      for(j = i; j >= gap && (data[j - gap] > temp); j -= gap) 
        data[j] = data[j - gap];

      data[j] = temp;
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
    bool interleaved) {

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
    sh_idx = th_idx * dim + i;
    
    if(interleaved)
      global_idx = index + th_idx * dim + i;
    else
      global_idx = index + th_idx + count * i;

    real arg = d_arg[global_idx];

    // TODO: check if this should be plus or minus
    if(d_coeffs != NULL)
      arg += (1. / tau) * d_coeffs[global_idx];
    
    sh_y[sh_idx] = arg;
  }
  __syncthreads();
  
  // 2) sort inside shared memory
  shellsort<real>(&sh_y[th_idx * dim], dim);
  
  // 3/4) do computation
  i = dim - 1;

  real t_hat = 0;
  do {
    for(int j = i; j < dim; j++) {
      t_hat += sh_y[th_idx * dim + j];
    }
    t_hat = (t_hat - 1) / (real)(dim - i);

    // if t_i >= y_i we're done!
    if(t_hat >= sh_y[th_idx * dim + i])
      break;

    i--;
  } while(i >= 0);

  // 5) return result
  for(i = 0; i < dim; i++) {
    if(interleaved)
      global_idx = index + th_idx * dim + i;
    else
      global_idx = index + th_idx + count * i;

    d_result[global_idx] = cuwrap::max<real>(sh_y[th_idx * dim + i] - t_hat, 0);
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

  size_t shmem_bytes = dim_ * kBlockSizeCUDA * sizeof(real);

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
          interleaved_);
  
  cudaDeviceSynchronize();
}

int ProxSimplex::gpu_mem_amount() {
  if(d_coeffs_ != 0)
    return count_ * sizeof(real);
  else
    return 0;
}
