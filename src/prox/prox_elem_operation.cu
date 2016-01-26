#include "prox/prox_elem_operation.hpp"

#include "config.hpp"
#include "util/cuwrap.hpp"

#include <iostream>

using namespace std;
using namespace thrust;
using namespace prox;
using namespace elemop;

template<typename T, class ELEM_OPERATION, class ENABLE = typename std::enable_if<!has_coeffs<ELEM_OPERATION>::value>::type>
__global__
void ProxElemOperationKernel(
    T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau,
    size_t count,
    bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    Vector<T, ELEM_OPERATION> res(count, interleaved, tx, d_res);
    Vector<T, ELEM_OPERATION> arg(count, interleaved, tx, d_arg);
    Vector<T, ELEM_OPERATION> tau_diag(count, interleaved, tx, d_tau);


    SharedMem<ELEM_OPERATION> sh_mem(threadIdx.x);

    ELEM_OPERATION op;
    op(arg, res, tau_diag, tau, invert_tau, sh_mem);
  }
}


template<typename T, class ELEM_OPERATION, class ENABLE = typename std::enable_if<has_coeffs<ELEM_OPERATION>::value>::type>
__global__
void ProxElemOperationKernel(
    T *d_arg,
    T *d_res,
    T *d_tau,
    T tau,
    bool invert_tau,
    typename ELEM_OPERATION::Coefficients* d_coeffs,
    size_t count,
    bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    typename ELEM_OPERATION::Coefficients coeffs = d_coeffs[tx];
    
    Vector<T, ELEM_OPERATION> res(count, interleaved, tx, d_res);
    Vector<T, ELEM_OPERATION> arg(count, interleaved, tx, d_arg);
    Vector<T, ELEM_OPERATION> tau_diag(count, interleaved, tx, d_tau);


    SharedMem<ELEM_OPERATION> sh_mem(threadIdx.x);

    ELEM_OPERATION op(coeffs);
    op(arg, res, tau_diag, tau, invert_tau, sh_mem);
  }
}

template<typename T, class ELEM_OPERATION>
void ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<!has_coeffs<ELEM_OPERATION>::value>::type>::EvalLocal(typename thrust::device_vector<T>::iterator d_arg_begin,
                         typename thrust::device_vector<T>::iterator d_arg_end,
                         typename thrust::device_vector<T>::iterator d_res_begin,
                         typename thrust::device_vector<T>::iterator d_res_end,
                         typename thrust::device_vector<T>::iterator d_tau_begin,
                         typename thrust::device_vector<T>::iterator d_tau_end,
                         T tau,
                         bool invert_tau) {
      dim3 block(kBlockSizeCUDA, 1, 1);
	  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

	  size_t shmem_bytes = ELEM_OPERATION::shared_mem_count * block.x * sizeof(typename ELEM_OPERATION::shared_mem_type);
      ProxElemOperationKernel<T, ELEM_OPERATION>
            <<<grid, block, shmem_bytes>>>(
             thrust::raw_pointer_cast(&(*d_arg_begin)),
             thrust::raw_pointer_cast(&(*d_res_begin)),
             thrust::raw_pointer_cast(&(*d_tau_begin)),
             tau,
             invert_tau,
             this->count_,
             this->interleaved_);
  }

template<typename T, class ELEM_OPERATION>
void ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<has_coeffs<ELEM_OPERATION>::value>::type>::EvalLocal(typename thrust::device_vector<T>::iterator d_arg_begin,
                         typename thrust::device_vector<T>::iterator d_arg_end,
                         typename thrust::device_vector<T>::iterator d_res_begin,
                         typename thrust::device_vector<T>::iterator d_res_end,
                         typename thrust::device_vector<T>::iterator d_tau_begin,
                         typename thrust::device_vector<T>::iterator d_tau_end,
                         T tau,
                         bool invert_tau) {
      dim3 block(kBlockSizeCUDA, 1, 1);
	  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

	  size_t shmem_bytes = ELEM_OPERATION::shared_mem_count * block.x * sizeof(typename ELEM_OPERATION::shared_mem_type);
	  ProxElemOperationKernel<T, ELEM_OPERATION>
            <<<grid, block, shmem_bytes>>>(
             thrust::raw_pointer_cast(&(*d_arg_begin)),
             thrust::raw_pointer_cast(&(*d_res_begin)),
             thrust::raw_pointer_cast(&(*d_tau_begin)),
             tau,
             invert_tau,
             raw_pointer_cast(&d_coeffs_[0]),
             this->count_,
             this->interleaved_);
  }

// Explicit template instantiation
template class ProxElemOperation<float, ElemOperation1D<float, Function1DZero<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, 7, Function1DZero<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DAbs<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, 7, Function1DHuber<float>>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 7>>;
