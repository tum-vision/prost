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

#ifndef PROST_DUAL_LINEAROPERATOR_HPP_
#define PROST_DUAL_LINEAROPERATOR_HPP_

#include "prost/linop/linearoperator.hpp"

namespace prost {

/// 
/// \brief Evaluates the negative transpose of the child linear operator.
/// 
template<typename T>
class DualLinearOperator : public LinearOperator<T> {
 public:
  DualLinearOperator(shared_ptr<LinearOperator<T>> child);
  virtual ~DualLinearOperator();

  virtual void Eval(
    device_vector<T>& result, 
    const device_vector<T>& rhs,
    T beta = 0);

  virtual void EvalAdjoint(
    device_vector<T>& result, 
    const device_vector<T>& rhs,
    T beta = 0);
  
    /// \brief Returns \sum_{col=1}^{ncols} |K_{row,col}|^{\alpha}.
  virtual T row_sum(size_t row, T alpha) const;

  /// \brief Returns \sum_{row=1}^{nrows} |K_{row,col}|^{\alpha}.
  virtual T col_sum(size_t col, T alpha) const;

  virtual size_t nrows() const;
  virtual size_t ncols() const;

  virtual size_t gpu_mem_amount() const { return child_->gpu_mem_amount(); }

 protected:
  shared_ptr<LinearOperator<T>> child_;
};

}

#endif // PROST_DUAL_LINEAROPERATOR_HPP_
