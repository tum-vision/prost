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

#ifndef PROST_BLOCK_SPARSE_HPP_
#define PROST_BLOCK_SPARSE_HPP_

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "prost/linop/block.hpp"

namespace prost {

/// 
/// \brief Linear operator based on sparse matrix.
/// 
template<typename T>
class BlockSparse : public Block<T> {
  BlockSparse(size_t row, size_t col, size_t nrows, size_t ncols);

public: 
  // TODO: add check somewhere if int32_t index is big enough
  static BlockSparse<T> *CreateFromCSC(
    size_t row,
    size_t col,
    int m,
    int n,
    int nnz,
    const vector<T>& val,
    const vector<int32_t>& ptr,
    const vector<int32_t>& ind);

  virtual ~BlockSparse();

  virtual void Initialize();

  /// \brief Required for preconditioners, row and col are "local" 
  ///        for the operator, which means they start at 0.
  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;

  virtual size_t gpu_mem_amount() const;
  
protected:
  // TODO: implement sparse matrix multiplication on CPU
  virtual void EvalLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);

  virtual void EvalAdjointLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);

  /// \brief Number of non-zero elements.
  size_t nnz_;

  static cusparseHandle_t cusp_handle_;
  cusparseMatDescr_t descr_;

  device_vector<int32_t> ind_, ind_t_;
  device_vector<int32_t> ptr_, ptr_t_;
  device_vector<T> val_, val_t_;

  vector<int32_t> host_ind_, host_ind_t_;
  vector<int32_t> host_ptr_, host_ptr_t_;
  vector<T> host_val_, host_val_t_;
};

} // namespace prost

#endif // PROST_BLOCK_SPARSE_HPP_
