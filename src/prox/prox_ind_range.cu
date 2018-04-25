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

#include "prost/prox/prox_ind_range.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {
  
  template<typename T>
  void ProxIndRange<T>::setA(int m,
			     int n,
			     int nnz,
			     const vector<T>& val,
			     const vector<int32_t>& ptr,
			     const vector<int32_t>& ind)
  {
    nrows_ = m;
    ncols_ = n;
    nnz_ = nnz;
    
    // create data on host
    host_ind_t_ = ind; 
    host_ptr_t_ = ptr; 
    host_val_t_ = val; 

    host_ind_.resize(nnz_);
    host_val_.resize(nnz_);
    host_ptr_.resize(nrows_ + 1);

    csr2csc(ncols_, 
	    nrows_, 
	    nnz_, 
	    &host_val_t_[0],
	    &host_ind_t_[0],
	    &host_ptr_t_[0],
	    &host_val_[0],
	    &host_ind_[0],
	    &host_ptr_[0]);
  }

  template<typename T>
  void ProxIndRange<T>::setAA(int m,
			      int n,
			      const vector<T>& val)
  {
    if(m != n)
      throw Exception("ProxIndRange: Matrix 'AA' must be square!");

    if(m != ncols_)
      throw Exception("ProxIndRange: Matrix 'AA' must fit dimension of 'A'!");
        
    host_AA_val_ = val;
  }

  template<>
  void ProxIndRange<double>::Initialize()
  {
    cusparseCreate(&cusp_handle_);
    cusolverDnCreate(&cusolver_handle_);
    
    cusparseCreateMatDescr(&descr_);
    cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

    // forward
    ind_.resize(nnz_);
    val_.resize(nnz_);
    ptr_.resize(nrows_ + 1);

    // transpose
    ind_t_.resize(nnz_);
    val_t_.resize(nnz_);
    ptr_t_.resize(ncols_ + 1);

    // copy to GPU
    thrust::copy(host_ind_t_.begin(), host_ind_t_.end(), ind_t_.begin());
    thrust::copy(host_ptr_t_.begin(), host_ptr_t_.end(), ptr_t_.begin());
    thrust::copy(host_val_t_.begin(), host_val_t_.end(), val_t_.begin());

    thrust::copy(host_ind_.begin(), host_ind_.end(), ind_.begin());
    thrust::copy(host_ptr_.begin(), host_ptr_.end(), ptr_.begin());
    thrust::copy(host_val_.begin(), host_val_.end(), val_.begin());

    // factorize AA
    AA_fac_.resize(ncols_ * ncols_);
    AA_val_.resize(ncols_ * ncols_);
    thrust::copy(host_AA_val_.begin(), host_AA_val_.end(), AA_fac_.begin());
    thrust::copy(host_AA_val_.begin(), host_AA_val_.end(), AA_val_.begin());

    int bufferSize = 0;
    cusolverDnDpotrf_bufferSize(cusolver_handle_,
				CUBLAS_FILL_MODE_LOWER,
				ncols_,
				thrust::raw_pointer_cast(AA_val_.data()),
				ncols_,
				&bufferSize);

    info_.resize(1);
    buffer_.resize(bufferSize);

    cusolverDnDpotrf(cusolver_handle_,
		     CUBLAS_FILL_MODE_LOWER,
		     ncols_,
		     thrust::raw_pointer_cast(AA_fac_.data()),
		     ncols_,
		     thrust::raw_pointer_cast(buffer_.data()),
		     bufferSize,
		     thrust::raw_pointer_cast(info_.data()));
          
    temp_.resize(ncols_);
  }
  
  template<>
  void ProxIndRange<float>::Initialize()
  {
    cusparseCreate(&cusp_handle_);
    cusolverDnCreate(&cusolver_handle_);
    
    cusparseCreateMatDescr(&descr_);
    cusparseSetMatType(descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(descr_, CUSPARSE_INDEX_BASE_ZERO);

    // forward
    ind_.resize(nnz_);
    val_.resize(nnz_);
    ptr_.resize(nrows_ + 1);

    // transpose
    ind_t_.resize(nnz_);
    val_t_.resize(nnz_);
    ptr_t_.resize(ncols_ + 1);

    // copy to GPU
    thrust::copy(host_ind_t_.begin(), host_ind_t_.end(), ind_t_.begin());
    thrust::copy(host_ptr_t_.begin(), host_ptr_t_.end(), ptr_t_.begin());
    thrust::copy(host_val_t_.begin(), host_val_t_.end(), val_t_.begin());

    thrust::copy(host_ind_.begin(), host_ind_.end(), ind_.begin());
    thrust::copy(host_ptr_.begin(), host_ptr_.end(), ptr_.begin());
    thrust::copy(host_val_.begin(), host_val_.end(), val_.begin());

    // factorize AA
    AA_fac_.resize(ncols_ * ncols_);
    AA_val_.resize(ncols_ * ncols_);
    thrust::copy(host_AA_val_.begin(), host_AA_val_.end(), AA_fac_.begin());
    thrust::copy(host_AA_val_.begin(), host_AA_val_.end(), AA_val_.begin());

    int bufferSize = 0;
    cusolverDnSpotrf_bufferSize(cusolver_handle_,
				CUBLAS_FILL_MODE_LOWER,
				ncols_,
				thrust::raw_pointer_cast(AA_val_.data()),
				ncols_,
				&bufferSize);

    info_.resize(1);
    buffer_.resize(bufferSize);

    cusolverDnSpotrf(cusolver_handle_,
		     CUBLAS_FILL_MODE_LOWER,
		     ncols_,
		     thrust::raw_pointer_cast(AA_fac_.data()),
		     ncols_,
		     thrust::raw_pointer_cast(buffer_.data()),
		     bufferSize,
		     thrust::raw_pointer_cast(info_.data()));
          
    temp_.resize(ncols_);
  }

  template<typename T>
  size_t ProxIndRange<T>::gpu_mem_amount() const
  {
    return 0; // TODO: implement me
  }
   
  template<>
  void ProxIndRange<float>::EvalLocal(
    const typename thrust::device_vector<float>::iterator& result_beg,
    const typename thrust::device_vector<float>::iterator& result_end,
    const typename thrust::device_vector<float>::const_iterator& arg_beg,
    const typename thrust::device_vector<float>::const_iterator& arg_end,
    const typename thrust::device_vector<float>::const_iterator& tau_beg,
    const typename thrust::device_vector<float>::const_iterator& tau_end,
    float tau,
    bool invert_tau)
  {
    const float alpha = 1;
    const float beta = 0;

    // apply A'
    cusparseScsrmv(cusp_handle_,
		   CUSPARSE_OPERATION_NON_TRANSPOSE,
		   ncols_,
		   nrows_,
		   nnz_,
		   &alpha,
		   descr_,
		   thrust::raw_pointer_cast(val_t_.data()),
		   thrust::raw_pointer_cast(ptr_t_.data()),
		   thrust::raw_pointer_cast(ind_t_.data()),
		   thrust::raw_pointer_cast(&(*arg_beg)),
		   &beta,
		   thrust::raw_pointer_cast(temp_.data()));
    
  // solve system
    cusolverDnSpotrs(cusolver_handle_,
		     CUBLAS_FILL_MODE_LOWER,
		     temp_.size(),
		     1,
		     thrust::raw_pointer_cast(AA_fac_.data()),
		     temp_.size(),
		     thrust::raw_pointer_cast(temp_.data()),
		     temp_.size(),
		     thrust::raw_pointer_cast(info_.data()));
      
    // apply A
    cusparseScsrmv(cusp_handle_,
		   CUSPARSE_OPERATION_NON_TRANSPOSE,
		   nrows_,
		   ncols_,
		   nnz_,
		   &alpha,
		   descr_,
		   thrust::raw_pointer_cast(val_.data()),
		   thrust::raw_pointer_cast(ptr_.data()),
		   thrust::raw_pointer_cast(ind_.data()),
		   thrust::raw_pointer_cast(temp_.data()),
		   &beta,
		   thrust::raw_pointer_cast(&(*result_beg)));
  }

  template<>
  void ProxIndRange<double>::EvalLocal(
    const typename thrust::device_vector<double>::iterator& result_beg,
    const typename thrust::device_vector<double>::iterator& result_end,
    const typename thrust::device_vector<double>::const_iterator& arg_beg,
    const typename thrust::device_vector<double>::const_iterator& arg_end,
    const typename thrust::device_vector<double>::const_iterator& tau_beg,
    const typename thrust::device_vector<double>::const_iterator& tau_end,
    double tau,
    bool invert_tau)
  {
    const double alpha = 1;
    const double beta = 0;

    // apply A'
    cusparseDcsrmv(cusp_handle_,
		   CUSPARSE_OPERATION_NON_TRANSPOSE,
		   ncols_,
		   nrows_,
		   nnz_,
		   &alpha,
		   descr_,
		   thrust::raw_pointer_cast(val_t_.data()),
		   thrust::raw_pointer_cast(ptr_t_.data()),
		   thrust::raw_pointer_cast(ind_t_.data()),
		   thrust::raw_pointer_cast(&(*arg_beg)),
		   &beta,
		   thrust::raw_pointer_cast(temp_.data()));
    
    // solve system
    cusolverDnDpotrs(cusolver_handle_,
		     CUBLAS_FILL_MODE_LOWER,
		     temp_.size(),
		     1,
		     thrust::raw_pointer_cast(AA_fac_.data()),
		     temp_.size(),
		     thrust::raw_pointer_cast(temp_.data()),
		     temp_.size(),
		     thrust::raw_pointer_cast(info_.data()));
      
    // apply A
    cusparseDcsrmv(cusp_handle_,
		   CUSPARSE_OPERATION_NON_TRANSPOSE,
		   nrows_,
		   ncols_,
		   nnz_,
		   &alpha,
		   descr_,
		   thrust::raw_pointer_cast(val_.data()),
		   thrust::raw_pointer_cast(ptr_.data()),
		   thrust::raw_pointer_cast(ind_.data()),
		   thrust::raw_pointer_cast(temp_.data()),
		   &beta,
		   thrust::raw_pointer_cast(&(*result_beg)));
  }

  // Explicit template instantiation
  template class ProxIndRange<float>;
  template class ProxIndRange<double>;

} // namespace prost