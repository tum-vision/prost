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

#ifndef PROST_BLOCK_ID_KRON_DENSE_HPP_
#define PROST_BLOCK_ID_KRON_DENSE_HPP_

#include "prost/linop/block.hpp"

namespace prost {

///
/// \brief Linear operator composed of many small dense matrices.
///        Implements linear operator for kron(speye(diaglength), M),
///        where M is a small sparse matrix.
///
/// TODO: * add option to explicitly store transpose (less memory efficient
///         but faster)
///       * compare against running cublas*gemv in parallel
template<typename T>
class BlockIdKronDense : public Block<T>
{
  BlockIdKronDense(size_t row, size_t col, size_t nrows, size_t ncols);

public:
  static BlockIdKronDense<T> *CreateFromColFirstData(
    size_t diaglength,
    size_t row,
    size_t col,
    size_t nrows,
    size_t ncols,
    const std::vector<T>& data);

  virtual ~BlockIdKronDense() {}

  virtual void Initialize();

  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;

  virtual size_t gpu_mem_amount() const;

protected:
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

private:
  /// \brief Size of diagonal identity matrix Id for kron(M, Id).
  size_t diaglength_;

  /// \brief Number of rows in small dense matrix M.
  size_t mat_nrows_;

  /// \brief Number of columns in small dense matrix M.
  size_t mat_ncols_;

  /// \brief GPU/CPU data for dense matrix M
  device_vector<T> data_;
  vector<T> host_data_;
};

} // namespace prost

#endif //PROST_BLOCK_ID_KRON_DENSE_HPP_
