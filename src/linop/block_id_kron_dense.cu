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

#include "prost/linop/block_id_kron_dense.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost
{

template<typename T, bool transpose>
__global__ void BlockIdKronDenseKernel(
    T *result,
    const T *rhs,
    size_t diaglength,
    size_t nrows,
    size_t ncols,
    const T *data)
{
  const size_t tx = threadIdx.x + blockIdx.x * blockDim.x;

  if(!transpose) {
    if(tx < diaglength * nrows) {
      size_t row = tx % nrows;
      size_t col_ofs = (tx / nrows) * ncols;

      T sum = 0;
      for(int32_t i = 0; i < ncols; i++) {
	sum += data[i * nrows + row] * rhs[col_ofs + i];
      }

      result[tx] += sum;
    }
  }
  else {
    if(tx < diaglength * ncols) {
      size_t col = tx % ncols;
      size_t row_ofs = (tx / ncols) * nrows;

      T sum = 0;
      for(int32_t i = 0; i < nrows; i++) {
	sum += data[i + col * nrows] * rhs[row_ofs + i];
      }

      result[tx] += sum;
    }
  }
}

template<typename T>
BlockIdKronDense<T>::BlockIdKronDense(size_t row, size_t col, size_t nrows, size_t ncols)
    : Block<T>(row, col, nrows, ncols)
{
}

template<typename T>
BlockIdKronDense<T> *BlockIdKronDense<T>::CreateFromColFirstData(
    size_t diaglength,
    size_t row,
    size_t col,
    size_t nrows,
    size_t ncols,
    const std::vector<T>& data)
{
  BlockIdKronDense<T> *block = new BlockIdKronDense<T>(row, col, ((size_t)nrows) * diaglength, ((size_t)ncols) * diaglength);

  block->diaglength_ = diaglength;
  block->mat_nrows_ = nrows;
  block->mat_ncols_ = ncols;
  block->host_data_ = data;

  return block;
}

template<typename T>
void BlockIdKronDense<T>::Initialize()
{
  data_.resize(this->mat_nrows_ * this->mat_ncols_);
  thrust::copy(host_data_.begin(), host_data_.end(), data_.begin());
}

template<typename T>
T BlockIdKronDense<T>::row_sum(size_t row, T alpha) const
{
  row = row % mat_nrows_;
  
  T sum = 0;
  for(int32_t i = 0; i < mat_ncols_; i++)
    sum += std::pow(std::abs(host_data_[i * mat_nrows_ + row]), alpha);

  return sum;
}

template<typename T>
T BlockIdKronDense<T>::col_sum(size_t col, T alpha) const
{
  col = col % mat_ncols_;
  
  T sum = 0;
  for(int32_t i = 0; i < mat_nrows_; i++)
    sum += std::pow(std::abs(host_data_[i + col * mat_nrows_]), alpha);

  return sum;
}

template<typename T>
size_t BlockIdKronDense<T>::gpu_mem_amount() const
{
  return host_data_.size() * sizeof(T);
}

template<typename T>
void BlockIdKronDense<T>::EvalLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->nrows() + block.x) / block.x, 1, 1);

  BlockIdKronDenseKernel<T, false>
      <<<grid, block>>>(
          thrust::raw_pointer_cast(&(*res_begin)),
          thrust::raw_pointer_cast(&(*rhs_begin)),
          diaglength_,
          mat_nrows_,
          mat_ncols_,
          thrust::raw_pointer_cast(data_.data()));
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "BlockIdKronDense (forward): CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

template<typename T>
void BlockIdKronDense<T>::EvalAdjointLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->ncols() + block.x) / block.x, 1, 1);

  BlockIdKronDenseKernel<T, true>
      <<<grid, block>>>(
          thrust::raw_pointer_cast(&(*res_begin)),
          thrust::raw_pointer_cast(&(*rhs_begin)),
          diaglength_,
          mat_nrows_,
          mat_ncols_,
          thrust::raw_pointer_cast(data_.data()));
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "BlockIdKronDense (forward): CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

// Explicit template instantiation
template class BlockIdKronDense<float>;
template class BlockIdKronDense<double>;

} // namespace prost