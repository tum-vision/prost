#include "linop/linop_identity.hpp"

#include "config.hpp"
#include <cuda_runtime.h>

static const size_t LINOP_IDENTITY_CMEM_SIZE = 1024;

__constant__ float cmem_factors[LINOP_IDENTITY_CMEM_SIZE];
__constant__ size_t cmem_offsets[LINOP_IDENTITY_CMEM_SIZE];

template<> size_t LinOpIdentity<float>::cmem_counter_ = 0;
template<> size_t LinOpIdentity<double>::cmem_counter_ = 0;

template<typename T>
__global__
void LinOpIdentityKernel(T *d_res, T *d_rhs, size_t ndiags, size_t nrows, size_t ncols, size_t cmem_idx) {
  size_t row = threadIdx.x + blockIdx.x * blockDim.x;

  T result = 0;
  for(size_t i = 0; i < ndiags; i++) {
    const size_t col = row + cmem_offsets[cmem_idx + i];

    if(col >= ncols)
      break;

    result += d_rhs[col] * cmem_factors[cmem_idx + i];
  }

  d_res[row] = result;
}

template<typename T>
__global__
void LinOpIdentityAdjointKernel(T *d_res, T *d_rhs, size_t ndiags, size_t nrows, size_t ncols, size_t cmem_idx) {
  size_t col = threadIdx.x + blockIdx.x * blockDim.x;

  T result = 0;
  for(size_t i = 0; i < ndiags; i++) {
    size_t ofs = cmem_offsets[cmem_idx + i];
    
    if(ofs <= col && (col - ofs) < nrows) {
      result += d_rhs[col - ofs] * cmem_factors[cmem_idx + i];
    }

    if(ofs > col)
      break;
  }

  d_res[col] = result;
}

template<typename T>
LinOpIdentity<T>::LinOpIdentity(size_t row,
                                size_t col,
                                size_t nrows,
                                size_t ncols,
                                size_t ndiags,
                                const std::vector<size_t>& offsets,
                                const std::vector<T>& factors)
    : LinOp<T>(row, col, nrows, ncols), cmem_offset_(0), ndiags_(ndiags), offsets_(offsets)
{
  for(size_t i = 0; i < factors.size(); i++)
    factors_[i] = static_cast<float>(factors[i]);
}

template<typename T>
LinOpIdentity<T>::~LinOpIdentity() {
  Release();
}

template<typename T>
bool LinOpIdentity<T>::Init() {
  cmem_offset_ = cmem_counter_;
  cmem_counter_ += ndiags_;

  if(cmem_counter_ >= LINOP_IDENTITY_CMEM_SIZE)
    return false;

  cudaMemcpyToSymbol(cmem_factors, &factors_[0], sizeof(float) * ndiags_);
  cudaMemcpyToSymbol(cmem_offsets, &offsets_[0], sizeof(size_t) * ndiags_);
  
  return true;
}

template<typename T>
void LinOpIdentity<T>::Release() {
}

template<typename T>
void LinOpIdentity<T>::EvalLocalAdd(T *d_res, T *d_rhs) {
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->nrows_ + block.x - 1) / block.x, 1, 1);

  LinOpIdentityKernel<T>
      <<<grid, block>>>(d_res,
                        d_rhs,
                        ndiags_,
                        this->nrows_,
                        this->ncols_,
                        cmem_offset_);
}

template<typename T>
void LinOpIdentity<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->ncols_ + block.x - 1) / block.x, 1, 1);

  LinOpIdentityAdjointKernel<T>
      <<<grid, block>>>(d_res,
                        d_rhs,
                        ndiags_,
                        this->nrows_,
                        this->ncols_,
                        cmem_offset_);
}

template<typename T>
T LinOpIdentity<T>::row_sum(size_t row, T alpha) const {
  T sum = 0;

  for(size_t i = 0; i < ndiags_; i++) {
    const size_t col = row + offsets_[i];

    if(col >= this->ncols_)
      break;

    sum += factors_[i];
  }

  return sum;
}

template<typename T>
T LinOpIdentity<T>::col_sum(size_t col, T alpha) const {
  T sum = 0;
  for(size_t i = 0; i < ndiags_; i++) {
    size_t ofs = offsets_[i];
    
    if(ofs <= col && (col - ofs) < this->nrows_) {
      sum += factors_[i];
    }

    if(ofs > col)
      break;
  }

  return sum;
}

// Explicit template instantiation
template class LinOpIdentity<float>;
template class LinOpIdentity<double>;
