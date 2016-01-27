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
    size_t dim,
    bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    Vector<T, ELEM_OPERATION> res(count, interleaved, tx, d_res);
    Vector<T, ELEM_OPERATION> arg(count, interleaved, tx, d_arg);
    Vector<T, ELEM_OPERATION> tau_diag(count, interleaved, tx, d_tau);


    SharedMem<ELEM_OPERATION> sh_mem(threadIdx.x);

    ELEM_OPERATION op(dim, sh_mem);
    op(arg, res, tau_diag, tau, invert_tau);
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
    size_t dim,
    bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    typename ELEM_OPERATION::Coefficients coeffs = d_coeffs[tx];
    
    Vector<T, ELEM_OPERATION> res(count, interleaved, tx, d_res);
    Vector<T, ELEM_OPERATION> arg(count, interleaved, tx, d_arg);
    Vector<T, ELEM_OPERATION> tau_diag(count, interleaved, tx, d_tau);


    SharedMem<ELEM_OPERATION> sh_mem(threadIdx.x);

    ELEM_OPERATION op(coeffs, dim, sh_mem);
    op(arg, res, tau_diag, tau, invert_tau);
  }
}

template<typename T, class ELEM_OPERATION>
void ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<!has_coeffs<ELEM_OPERATION>::value>::type>::EvalLocal(
                         const typename thrust::device_vector<T>::iterator& arg_begin,
                         const typename thrust::device_vector<T>::iterator& arg_end,
                         const typename thrust::device_vector<T>::iterator& res_begin,
                         const typename thrust::device_vector<T>::iterator& res_end,
                         const typename thrust::device_vector<T>::iterator& tau_begin,
                         const typename thrust::device_vector<T>::iterator& tau_end,
                         T tau,
                         bool invert_tau) {
      dim3 block(kBlockSizeCUDA, 1, 1);
	  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

	  size_t shmem_bytes = ELEM_OPERATION::shared_mem_count * block.x * sizeof(typename ELEM_OPERATION::shared_mem_type);
      ProxElemOperationKernel<T, ELEM_OPERATION>
            <<<grid, block, shmem_bytes>>>(
             thrust::raw_pointer_cast(&(*arg_begin)),
             thrust::raw_pointer_cast(&(*res_begin)),
             thrust::raw_pointer_cast(&(*tau_begin)),
             tau,
             invert_tau,
             this->count_,
             this->dim_,
             this->interleaved_);
  }

template<typename T, class ELEM_OPERATION>
void ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<has_coeffs<ELEM_OPERATION>::value>::type>::EvalLocal(
                         const typename thrust::device_vector<T>::iterator& arg_begin,
                         const typename thrust::device_vector<T>::iterator& arg_end,
                         const typename thrust::device_vector<T>::iterator& res_begin,
                         const typename thrust::device_vector<T>::iterator& res_end,
                         const typename thrust::device_vector<T>::iterator& tau_begin,
                         const typename thrust::device_vector<T>::iterator& tau_end,
                         T tau,
                         bool invert_tau) {
      dim3 block(kBlockSizeCUDA, 1, 1);
	  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

	  size_t shmem_bytes = ELEM_OPERATION::shared_mem_count * block.x * sizeof(typename ELEM_OPERATION::shared_mem_type);
	  ProxElemOperationKernel<T, ELEM_OPERATION>
            <<<grid, block, shmem_bytes>>>(
             thrust::raw_pointer_cast(&(*arg_begin)),
             thrust::raw_pointer_cast(&(*res_begin)),
             thrust::raw_pointer_cast(&(*tau_begin)),
             tau,
             invert_tau,
             raw_pointer_cast(&d_coeffs_[0]),
             this->count_,
             this->dim_,
             this->interleaved_);
  }



// Explicit template instantiation
template class ProxElemOperation<float, ElemOperation1D<float, Function1DZero<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DAbs<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DSquare<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndLeq0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndGeq0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndEq0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndBox01<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DMaxPos0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DL0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DHuber<float>>>;

template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DZero<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DAbs<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DSquare<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndLeq0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndGeq0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndEq0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndBox01<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DMaxPos0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DL0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DHuber<float>>>;

template class ProxElemOperation<float, ElemOperationSimplex<float, 2>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 3>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 4>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 5>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 6>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 7>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 8>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 9>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 10>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 11>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 12>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 13>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 14>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 15>>;
template class ProxElemOperation<float, ElemOperationSimplex<float, 16>>;

template class ProxElemOperation<double, ElemOperation1D<double, Function1DZero<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DAbs<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DSquare<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndLeq0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndGeq0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndEq0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndBox01<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DMaxPos0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DL0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DHuber<double>>>;

template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DZero<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DAbs<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DSquare<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndLeq0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndGeq0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndEq0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndBox01<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DMaxPos0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DL0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DHuber<double>>>;

template class ProxElemOperation<double, ElemOperationSimplex<double, 2>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 3>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 4>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 5>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 6>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 7>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 8>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 9>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 10>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 11>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 12>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 13>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 14>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 15>>;
template class ProxElemOperation<double, ElemOperationSimplex<double, 16>>;
