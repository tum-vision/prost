#include "prox/prox_elem_operation.hpp"

#include "config.hpp"
#include "util/cuwrap.hpp"

#include <iostream>

template<typename T, class ELEM_OPERATION>
__global__
void ProxElemOperationKernel(
    T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau,
    ELEM_OPERATION::Coefficients* d_coeffs,
    size_t count,
    bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    ELEM_OPERATION::Coefficients coeffs = d_coeffs[tx];
    
    Vector<T, ELEM_OPERATION>(count, interleaved, d_res, tx) res;
    Vector<T, ELEM_OPERATION>(count, interleaved, d_arg, tx) arg;
    Vector<T, ELEM_OPERATION>(count, interleaved, d_tau, tx) tau_diag;

    SharedMem<ELEM_OPERATION> sh_mem(threadIdx.x);

    ELEM_OPERATION op(coeffs);
    op(res, arg, tau_diag, tau, invert_tau, sh_mem);
  }
}

template<typename T, class ELEM_OPERATION>
ProxElemOperation<T, ELEM_OPERATION>::ProxElemOperation(size_t index, size_t count, interleaved, diagsteps, const vector<ELEM_OPERATION::Coefficients>& coeffs)
    : ProxSeparableSum<T>(index, count, ELEM_OPERATION::dim, interleaved, diagsteps) : coeffs_(coeffs)
{}

template<typename T, class ELEM_OPERATION>
ProxElemOperation<T, ELEM_OPERATION>::~ProxElemOperation() {
  Release();
}

template<typename T, class ELEM_OPERATION>
bool ProxElemOperation<T, ELEM_OPERATION>::Init() {
  copy(coeffs_.begin(), coeffs_.end(), d_coeffs_.begin());

  return true;
}

template<typename T, class ELEM_OPERATION>
void ProxElemOperation<T, ELEM_OPERATION>::Release() {

}

template<typename T, class ELEM_OPERATION>
void ProxElemOperation<T, ELEM_OPERATION>::EvalLocal(device_vector<T> d_arg,
                          device_vector<T> d_res,
                          device_vector<T> d_tau,
                          T tau,
                          bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  size_t shmem_bytes = ELEM_OPERATION::shared_mem_count * block.x * sizeof(ELEM_OPERATION::shared_mem_type);

  ProxElemOperationKernel<T, ELEM_OPERATION>
      <<<grid, block, shmem_bytes>>>(
             raw_pointer_cast(&d_arg[0]),
             raw_pointer_cast(&d_res[0]),
             raw_pointer_cast(&d_tau[0]),
             tau,
             invert_tau,
             raw_pointer_cast(&this->d_coeffs_[0]),
             this->count_,
             this->interleaved_);
}

template<typename T, class ELEM_OPERATION>
size_t ProxElemOperation<T, ELEM_OPERATION>::gpu_mem_amount() {
  // TODO correct??? Should add shared memory?
  size_t total_mem = this->count_ * sizeof(ELEM_OPERATION::Coefficients);

  return total_mem;
}

// Explicit template instantiation
template class ProxElemOperation<float>;
template class ProxElemOperation<double>;
