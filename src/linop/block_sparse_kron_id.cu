/**
* This file is part of prost.
*
* Copyright 2016 Thomas Möllenhoff <thomas dot moellenhoff at in dot tum dot de> 
* and Emanuel Laude <emanuel dot laude at in dot tum dot de> (Technical University of Munich)
*
* prost is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* prost is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with prost. If not, see <http://www.gnu.org/licenses/>.
*/

#include "prost/linop/block_sparse_kron_id.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
__global__ void BlockSparseKronIdKernel(
    T *result,
    const T *rhs,
    size_t diaglength,
    size_t nrows,
    const int32_t *ind,
    const int32_t *ptr,
    const float *val)
{
  const size_t tx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tx < diaglength * nrows)
  {
    size_t col_ofs = tx % diaglength;
    size_t row = tx / diaglength;

    T sum = 0;
    int32_t stop = ptr[row + 1];
    
    for(int32_t i = ptr[row]; i < stop; i++)
      sum += val[i] * rhs[ind[i] * diaglength + col_ofs];

    result[tx] += sum;
  }
}

template<typename T>
BlockSparseKronId<T>::BlockSparseKronId(size_t row, size_t col, size_t nrows, size_t ncols)
    : Block<T>(row, col, nrows, ncols)
{
}

template<typename T>
BlockSparseKronId<T> *BlockSparseKronId<T>::CreateFromCSC(
    size_t row,
    size_t col,
    size_t diaglength,
    int m,
    int n,
    int nnz,
    const vector<T>& val,
    const vector<int32_t>& ptr,
    const vector<int32_t>& ind)
{
  BlockSparseKronId<T> *block = new BlockSparseKronId<T>(row, col, ((size_t)m) * diaglength, ((size_t)n) * diaglength);

  block->diaglength_ = diaglength;
  block->mat_nnz_ = nnz;
  block->mat_nrows_ = m;
  block->mat_ncols_ = n;

  // create data on host
  block->host_ind_t_ = ind; 
  block->host_ptr_t_ = ptr; 
  block->host_val_t_ = std::vector<float>(val.begin(), val.end()); 

  block->host_ind_.resize(block->mat_nnz_);
  block->host_val_.resize(block->mat_nnz_);
  block->host_ptr_.resize(block->mat_nrows_ + 1);

  csr2csc(
    block->mat_ncols_, 
    block->mat_nrows_, 
    block->mat_nnz_, 
    &block->host_val_t_[0],
    &block->host_ind_t_[0],
    &block->host_ptr_t_[0],
    &block->host_val_[0],
    &block->host_ind_[0],
    &block->host_ptr_[0]);

  return block;
}

template<typename T>
void BlockSparseKronId<T>::Initialize()
{
  // forward
  ind_.resize(mat_nnz_);
  val_.resize(mat_nnz_);
  ptr_.resize(mat_nrows_ + 1);

  // transpose
  ind_t_.resize(mat_nnz_);
  val_t_.resize(mat_nnz_);
  ptr_t_.resize(mat_ncols_ + 1);

  // copy to GPU
  thrust::copy(host_ind_t_.begin(), host_ind_t_.end(), ind_t_.begin());
  thrust::copy(host_ptr_t_.begin(), host_ptr_t_.end(), ptr_t_.begin());
  thrust::copy(host_val_t_.begin(), host_val_t_.end(), val_t_.begin());

  thrust::copy(host_ind_.begin(), host_ind_.end(), ind_.begin());
  thrust::copy(host_ptr_.begin(), host_ptr_.end(), ptr_.begin());
  thrust::copy(host_val_.begin(), host_val_.end(), val_.begin());
}

template<typename T>
T BlockSparseKronId<T>::row_sum(size_t row, T alpha) const
{
  row = row / diaglength_;
  
  T sum = 0;

  for(int32_t i = host_ptr_[row]; i < host_ptr_[row + 1]; i++)
    sum += std::pow(std::abs(host_val_[i]), alpha);

  return sum;
}

template<typename T>
T BlockSparseKronId<T>::col_sum(size_t col, T alpha) const
{
  col = col / diaglength_;
  
  T sum = 0;

  for(int32_t i = host_ptr_t_[col]; i < host_ptr_t_[col + 1]; i++)
    sum += std::pow(std::abs(host_val_t_[i]), alpha);

  return sum;
}

template<typename T>
size_t BlockSparseKronId<T>::gpu_mem_amount() const
{
  return (host_ind_.size() + host_ind_t_.size() + host_ptr_.size() + host_ptr_t_.size()) * sizeof(int32_t) + (host_val_.size() + host_val_t_.size()) * sizeof(T);
}

template<typename T>
void BlockSparseKronId<T>::EvalLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->nrows() + block.x) / block.x, 1, 1);

  BlockSparseKronIdKernel<T>
      <<<grid, block>>>(
          thrust::raw_pointer_cast(&(*res_begin)),
          thrust::raw_pointer_cast(&(*rhs_begin)),
          diaglength_,
          mat_nrows_,
          thrust::raw_pointer_cast(ind_.data()),
          thrust::raw_pointer_cast(ptr_.data()),
          thrust::raw_pointer_cast(val_.data()));
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "BlockSparseKronId: CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

template<typename T>
void BlockSparseKronId<T>::EvalAdjointLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->ncols() + block.x) / block.x, 1, 1);

  BlockSparseKronIdKernel<T>
      <<<grid, block>>>(
          thrust::raw_pointer_cast(&(*res_begin)),
          thrust::raw_pointer_cast(&(*rhs_begin)),
          diaglength_,
          mat_ncols_,
          thrust::raw_pointer_cast(ind_t_.data()),
          thrust::raw_pointer_cast(ptr_t_.data()),
          thrust::raw_pointer_cast(val_t_.data()));
  cudaDeviceSynchronize();

  // check for error  
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "BlockSparseKronId: CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

// Explicit template instantiation
template class BlockSparseKronId<float>;
template class BlockSparseKronId<double>;

} // namespace prost