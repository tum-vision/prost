#include "prox/prox_elem_operation.hpp"
#include "config.hpp"
#include "exception.hpp"

using namespace std;
using namespace thrust;
using namespace prox;

template<typename T, class ELEM_OPERATION>
struct Coefficients 
{
  T* dev_p[ELEM_OPERATION::kCoeffsCount];
  T val[ELEM_OPERATION::kCoeffsCount];
};

template<typename T, class ELEM_OPERATION, class ENABLE = typename std::enable_if<ELEM_OPERATION::coeffs_count == 0>::type>
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

  if(tx < count) 
  {
    Vector<T> res(count, dim, interleaved, tx, d_res);
    Vector<T> arg(count, dim, interleaved, tx, d_arg);
    Vector<T> tau_diag(count, dim, interleaved, tx, d_tau);

    SharedMem<ELEM_OPERATION> sh_mem(dim, threadIdx.x);

    ELEM_OPERATION op(dim, sh_mem);
    op(arg, res, tau_diag, tau, invert_tau);
  }
}


template<typename T, class ELEM_OPERATION, class ENABLE = typename std::enable_if<ELEM_OPERATION::coeffs_count != 0>::type>
__global__
void ProxElemOperationKernel(
  T *d_arg,
  T *d_res,
  T *d_tau,
  T tau,
  bool invert_tau,
  size_t count,
  size_t dim,
  Coefficients<T, ELEM_OPERATION> coeffs,
  bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count) {
    Vector<T> res(count, dim, interleaved, tx, d_res);
    Vector<T> arg(count, dim, interleaved, tx, d_arg);
    Vector<T> tau_diag(count, dim, interleaved, tx, d_tau);

    SharedMem<ELEM_OPERATION> sh_mem(dim, threadIdx.x);

    T coeffs_local[ELEM_OPERATION::coeffs_count];
    for(int i = 0; i < ELEM_OPERATION::coeffs_count; i++) {
      if(coeffs.dev_p[i] == nullptr) {
        coeffs_local[i] = coeffs.val[i];
      } else {
        coeffs_local[i] = coeffs.dev_p[i][tx];
      }
    }

    ELEM_OPERATION op(coeffs_local, dim, sh_mem);
    op(arg, res, tau_diag, tau, invert_tau);
  }
}

template<typename T, class ELEM_OPERATION>
void 
ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::coeffs_count == 0>::type>::EvalLocal(
    const typename thrust::device_vector<T>::iterator& result_beg,
    const typename thrust::device_vector<T>::iterator& result_end,
    const typename thrust::device_vector<T>::const_iterator& arg_beg,
    const typename thrust::device_vector<T>::const_iterator& arg_end,
    const typename thrust::device_vector<T>::const_iterator& tau_beg,
    const typename thrust::device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  size_t shmem_bytes = ELEM_OPERATION::GetSharedMemCount(this->dim_) * block.x * sizeof(typename ELEM_OPERATION::SharedMemType);
  ProxElemOperationKernel<T, ELEM_OPERATION>
    <<<grid, block, shmem_bytes>>>(
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*tau_beg)),
      tau,
      invert_tau,
      this->count_,
      this->dim_,
      this->interleaved_);
}

template<typename T, class ELEM_OPERATION>
void 
ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::coeffs_count != 0>::type>::EvalLocal(
    const typename thrust::device_vector<T>::iterator& result_beg,
    const typename thrust::device_vector<T>::iterator& result_end,
    const typename thrust::device_vector<T>::const_iterator& arg_beg,
    const typename thrust::device_vector<T>::const_iterator& arg_end,
    const typename thrust::device_vector<T>::const_iterator& tau_beg,
    const typename thrust::device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  Coefficients<T, ELEM_OPERATION> coeffs;
     

  for(size_t i = 0; i < ELEM_OPERATION::coeffs_count; i++) {
    if(coeffs_[i].size() > 1) {
      coeffs.dev_p[i] = thrust::raw_pointer_cast(&d_coeffs_[i][0]);
    } else {
      coeffs.dev_p[i] = nullptr;
      coeffs.val[i] = coeffs_[i][0];
    }
  }

  size_t shmem_bytes = ELEM_OPERATION::shared_mem_count(this->dim_) * block.x * sizeof(typename ELEM_OPERATION::shared_mem_type);
  ProxElemOperationKernel<T, ELEM_OPERATION>
    <<<grid, block, shmem_bytes>>>(
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*tau_beg)),
      tau,
      invert_tau,
      this->count_,
      this->dim_,
      coeffs,
      this->interleaved_);
}

template<typename T, class ELEM_OPERATION>
void ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<ELEM_OPERATION::coeffs_count != 0>::type>::Init() 
{
  for(size_t i = 0; i < ELEM_OPERATION::kCoeffsCount; i++) { 
    try {
      if(coeffs_[i].size() > 1) {
        d_coeffs_[i].resize(coeffs_[i].size());
        thrust::copy(coeffs_[i].begin(), coeffs_[i].end(), d_coeffs_[i].begin());
      }
    } catch(std::bad_alloc &e) {
      throw Exception(e.what());
    } catch(thrust::system_error &e) {
      throw Exception(e.what());
    }
  }
}

// Explicit template instantiation
template class ProxElemOperation<float, ElemOperation1D<float, Function1DZero<float>>>;
/*
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

template class ProxElemOperation<float, ElemOperationSimplex<float>>;

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

template class ProxElemOperation<double, ElemOperationSimplex<double>>;
*/