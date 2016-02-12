#include "prost/linop/block_diags.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

// Maximum number of diagonals.
static const size_t kMaxNumberOfDiagonals = 1024; 

__constant__ float cmem_factors[kMaxNumberOfDiagonals];
__constant__ ssize_t cmem_offsets[kMaxNumberOfDiagonals];

template<> size_t BlockDiags<float>::cmem_counter_ = 0;
template<> size_t BlockDiags<double>::cmem_counter_ = 0;

template<typename T>
__global__
void BlockDiagsKernel(T *d_res,
		      const T *d_rhs,
		      size_t ndiags,
		      size_t nrows,
		      size_t ncols,
		      size_t cmem_idx)
{
  size_t row = threadIdx.x + blockIdx.x * blockDim.x;

  if(row >= nrows)
    return;

  T result = 0;
  for(size_t i = 0; i < ndiags; i++)
  {
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
void BlockDiagsAdjointKernel(T *d_res,
			     const T *d_rhs,
			     size_t ndiags,
			     size_t nrows,
			     size_t ncols,
			     size_t cmem_idx)
{
  ssize_t col = threadIdx.x + blockIdx.x * blockDim.x;

  if(col >= ncols)
    return;

  T result = 0;
  for(size_t i = 0; i < ndiags; i++)
  {
    ssize_t ofs = cmem_offsets[cmem_idx + i];
    
    if(ofs <= col && (col - ofs) < nrows && (col - ofs) >= 0)
    {
      result += d_rhs[col - ofs] * cmem_factors[cmem_idx + i];
    }

    if(ofs > col)
      break;
  }

  d_res[col] += result;
}

template<typename T>
BlockDiags<T>::BlockDiags(size_t row,
			  size_t col,
			  size_t nrows,
			  size_t ncols,
			  size_t ndiags,
			  const std::vector<ssize_t>& offsets,
			  const std::vector<T>& factors)
  : Block<T>(row, col, nrows, ncols), cmem_offset_(0), ndiags_(ndiags), offsets_(offsets)
{
  factors_ = std::vector<float>(factors.begin(), factors.end());

  // bubble sort, sort according to offsets 
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
T BlockDiags<T>::row_sum(size_t row, T alpha) const
{
  T sum = 0;

  for(size_t i = 0; i < ndiags_; i++) {
    const ssize_t col = row + offsets_[i];

    if(col < 0)
      continue;

    if(col >= this->ncols())
      break;

    sum += std::pow(std::abs(factors_[i]), alpha);
  }

  return sum;
}
  
template<typename T>
T BlockDiags<T>::col_sum(size_t col, T alpha) const
{
  T sum = 0;
  ssize_t signed_col = (ssize_t) col;

  for(size_t i = 0; i < ndiags_; i++) {
    ssize_t ofs = offsets_[i];
    
    if(ofs <= signed_col && 
       (signed_col - ofs) < (ssize_t)this->nrows() && 
      (signed_col - ofs) >= 0) 
    {
      sum += std::pow(std::abs(factors_[i]), alpha);
    }

    if(ofs > signed_col)
      break;
  }

  return sum;
}

template<typename T>
void BlockDiags<T>::Initialize()
{
  cmem_offset_ = cmem_counter_;
  cmem_counter_ += ndiags_;

  if(cmem_counter_ >= kMaxNumberOfDiagonals) 
  {
    throw Exception("Out of constant memory. Too many BlockDiags or too many diagonals.");
  }

  cudaMemcpyToSymbol(cmem_factors,
		     &factors_[0],
		     sizeof(float) * ndiags_,
		     cmem_offset_ * sizeof(float));
  
  cudaMemcpyToSymbol(cmem_offsets,
		     &offsets_[0],
		     sizeof(ssize_t) * ndiags_,
		     cmem_offset_ * sizeof(size_t)); 
}
  
template<typename T>
void BlockDiags<T>::EvalLocalAdd(const typename device_vector<T>::iterator& res_begin,
				 const typename device_vector<T>::iterator& res_end,
				 const typename device_vector<T>::const_iterator& rhs_begin,
				 const typename device_vector<T>::const_iterator& rhs_end)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->nrows() + block.x - 1) / block.x, 1, 1);

  BlockDiagsKernel<T>
    <<<grid, block>>>(thrust::raw_pointer_cast(&(*res_begin)),
		      thrust::raw_pointer_cast(&(*rhs_begin)),
		      ndiags_,
		      this->nrows(),
		      this->ncols(),
		      cmem_offset_);
}

template<typename T>
void BlockDiags<T>::EvalAdjointLocalAdd(const typename device_vector<T>::iterator& res_begin,
					const typename device_vector<T>::iterator& res_end,
					const typename device_vector<T>::const_iterator& rhs_begin,
					const typename device_vector<T>::const_iterator& rhs_end)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->nrows() + block.x - 1) / block.x, 1, 1);

  BlockDiagsAdjointKernel<T>
    <<<grid, block>>>(thrust::raw_pointer_cast(&(*res_begin)),
		      thrust::raw_pointer_cast(&(*rhs_begin)),
		      ndiags_,
		      this->nrows(),
		      this->ncols(),
		      cmem_offset_);
}

// Explicit template instantiation
template class BlockDiags<float>;
template class BlockDiags<double>;
  
}
