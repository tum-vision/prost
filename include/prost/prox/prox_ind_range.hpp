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

#ifndef PROST_PROX_IND_RANGE_
#define PROST_PROX_IND_RANGE_

#include <array>
#include <vector>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cusolverDn.h>

#include "prost/prox/prox.hpp"
#include "prost/prox/vector.hpp"
#include "prost/common.hpp"

namespace prost {

///
/// \brief Implements projection onto range of sparse matrix A. 
///        x = A * ((A' * A)^{-1} * (A' * x0))
///
template<typename T>
class ProxIndRange : public Prox<T> 
{
public:
  ProxIndRange(size_t index, 
	       size_t size, 
	       bool diagsteps) : Prox<T>(index, size, diagsteps) { }

  // call before initialize
  void setA(int m,
	    int n,
	    int nnz,
	    const vector<T>& val,
	    const vector<int32_t>& ptr,
	    const vector<int32_t>& ind);

  // call before initialize
  void setAA(int m,
	     int n,
	     const vector<T>& val);

  virtual void Initialize();
  
  virtual size_t gpu_mem_amount() const;
   
protected:
  virtual void EvalLocal(
    const typename thrust::device_vector<T>::iterator& result_beg,
    const typename thrust::device_vector<T>::iterator& result_end,
    const typename thrust::device_vector<T>::const_iterator& arg_beg,
    const typename thrust::device_vector<T>::const_iterator& arg_end,
    const typename thrust::device_vector<T>::const_iterator& tau_beg,
    const typename thrust::device_vector<T>::const_iterator& tau_end,
    T tau,
    bool invert_tau);
  
private:
  cusparseHandle_t cusp_handle_;
  cusolverDnHandle_t cusolver_handle_;
  cusparseMatDescr_t descr_;

  size_t nrows_, ncols_;
  size_t nnz_;

  vector<T> host_AA_val_;
  device_vector<T> buffer_, AA_fac_, AA_val_;
  device_vector<int> info_;

  device_vector<int32_t> ind_, ind_t_; 
  device_vector<int32_t> ptr_, ptr_t_; 
  device_vector<T> val_, val_t_; 
  device_vector<T> temp_;

  vector<int32_t> host_ind_, host_ind_t_; 
  vector<int32_t> host_ptr_, host_ptr_t_; 
  vector<T> host_val_, host_val_t_; 
};

} // namespace prost

#endif // PROST_PROX_IND_RANGE_
