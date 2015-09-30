#include "linop/linop_diags.hpp"

#include <cuda_runtime.h>
#include <iostream>
#include "config.hpp"
#include "util/util.hpp"

static const size_t LINOP_DIAGS_CMEM_SIZE = 1024;

__constant__ float cmem_factors[LINOP_DIAGS_CMEM_SIZE];
__constant__ ssize_t cmem_offsets[LINOP_DIAGS_CMEM_SIZE];

template<> size_t LinOpDiags<float>::cmem_counter_ = 0;
template<> size_t LinOpDiags<double>::cmem_counter_ = 0;

template<typename T>
__global__
void LinOpDiagsKernel(T *d_res, T *d_rhs, size_t ndiags, size_t nrows, size_t ncols, size_t cmem_idx) {
  size_t row = threadIdx.x + blockIdx.x * blockDim.x;

  if(row >= nrows)
    return;

  T result = 0;
  for(size_t i = 0; i < ndiags; i++) {
    const ssize_t col = row + cmem_offsets[cmem_idx + i];

    if(col < 0)
      continue;

    if(col >= ncols)
      break;

    result += d_rhs[col] * cmem_factors[cmem_idx + i];
  }

  d_res[row] += result;
}

template<typename T>
__global__
void LinOpDiagsAdjointKernel(T *d_res, T *d_rhs, size_t ndiags, size_t nrows, size_t ncols, size_t cmem_idx) {
  ssize_t col = threadIdx.x + blockIdx.x * blockDim.x;

  if(col >= ncols)
    return;

  T result = 0;
  for(size_t i = 0; i < ndiags; i++) {
    ssize_t ofs = cmem_offsets[cmem_idx + i];
    
    if(ofs <= col && (col - ofs) < nrows && (col - ofs) >= 0) {
      result += d_rhs[col - ofs] * cmem_factors[cmem_idx + i];
    }

    if(ofs > col)
      break;
  }

  d_res[col] += result;
}

template<typename T>
LinOpDiags<T>::LinOpDiags(size_t row,
  size_t col,
  size_t nrows,
  size_t ncols,
  size_t ndiags,
  const std::vector<ssize_t>& offsets,
  const std::vector<T>& factors)
  : LinOp<T>(row, col, nrows, ncols), cmem_offset_(0), ndiags_(ndiags), offsets_(offsets)
{
  for(size_t i = 0; i < factors.size(); i++) {
    factors_.push_back(static_cast<float>(factors[i]));
  }

  // bubble sort, sort offsets 
  for(size_t i = 0; i < factors.size(); i++) {
    for(size_t j = i; j < offsets.size(); j++) {
      if(offsets_[i] > offsets_[j]) {
        std::swap(offsets_[i], offsets_[j]);
        std::swap(factors_[i], factors_[j]);
      }
    }
  }

}

template<typename T>
LinOpDiags<T>::~LinOpDiags() {
  Release();
}

template<typename T>
bool LinOpDiags<T>::Init() {
  cmem_offset_ = cmem_counter_;
  cmem_counter_ += ndiags_;

  if(cmem_counter_ >= LINOP_DIAGS_CMEM_SIZE) 
  {
    std::cout << "Out of constant memory!" << std::endl;
    return false;
  }

  cudaMemcpyToSymbol(cmem_factors, &factors_[0], sizeof(float) * ndiags_, cmem_offset_ * sizeof(float)); 
  CUDA_CHECK;
  cudaMemcpyToSymbol(cmem_offsets, &offsets_[0], sizeof(ssize_t) * ndiags_, cmem_offset_ * sizeof(size_t)); 
  CUDA_CHECK;

  return true;
}

template<typename T>
void LinOpDiags<T>::Release() {
}

template<typename T>
void LinOpDiags<T>::EvalLocalAdd(T *d_res, T *d_rhs) {
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->nrows_ + block.x - 1) / block.x, 1, 1);

  LinOpDiagsKernel<T>
    <<<grid, block>>>(d_res,
      d_rhs,
      ndiags_,
      this->nrows_,
      this->ncols_,
      cmem_offset_);
  CUDA_CHECK;
}

template<typename T>
void LinOpDiags<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->ncols_ + block.x - 1) / block.x, 1, 1);

  LinOpDiagsAdjointKernel<T>
    <<<grid, block>>>(d_res,
      d_rhs,
      ndiags_,
      this->nrows_,
      this->ncols_,
      cmem_offset_);
  CUDA_CHECK;
}

template<typename T>
T LinOpDiags<T>::row_sum(size_t row, T alpha) const {
  T sum = 0;

  for(size_t i = 0; i < ndiags_; i++) {
    const ssize_t col = row + offsets_[i];

    if(col < 0)
      continue;

    if(col >= this->ncols_)
      break;

    sum += pow(abs(factors_[i]), alpha);
  }

  return sum;
}

template<typename T>
T LinOpDiags<T>::col_sum(size_t col, T alpha) const {
  T sum = 0;
  ssize_t signed_col = (ssize_t) col;

  for(size_t i = 0; i < ndiags_; i++) {
    ssize_t ofs = offsets_[i];
    
    if(ofs <= signed_col && 
      (signed_col - ofs) < (ssize_t)this->nrows_ && 
      (signed_col - ofs) >= 0) 
    {
      sum += pow(abs(factors_[i]), alpha);
    }

    if(ofs > signed_col)
      break;
  }

  return sum;
}

// Explicit template instantiation
template class LinOpDiags<float>;
template class LinOpDiags<double>;
