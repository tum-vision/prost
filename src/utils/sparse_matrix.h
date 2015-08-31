/*
 * This file is part of pdsolver.
 *
 * Copyright (C) 2015 Thomas Möllenhoff <thomas.moellenhoff@in.tum.de> 
 *
 * pdsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pdsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pdsolver. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef SPARSE_MATRIX_H
#define SPARSE_MATRIX_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cmath>
#include <cstdlib>

#include "cuwrapper.h"

template<typename real>
class sparse_matrix {
  sparse_matrix() { }
  
public:
  virtual ~sparse_matrix() {
    cudaFree(d_ind);
    cudaFree(d_val);
    cudaFree(d_ptr);
  }

  static sparse_matrix<real> *create_from_csc(int m,
                                              int n,
                                              int nnz,
                                              real *val,
                                              int *ptr,
                                              int *ind) {
    sparse_matrix<real> *mat = new sparse_matrix<real>;

    cusparseCreate(&mat->cusp_handle);
    cusparseCreateMatDescr(&mat->descr);
    cusparseSetMatType(mat->descr, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat->descr, CUSPARSE_INDEX_BASE_ZERO);

    mat->m = m;
    mat->n = n;
    mat->nnz = nnz;

    cudaMalloc((void **)&mat->d_ind, sizeof(int) * 2 * mat->nnz);
    cudaMalloc((void **)&mat->d_ptr, sizeof(int) * (mat->m + mat->n + 2));
    cudaMalloc((void **)&mat->d_val, sizeof(real) * 2 * mat->nnz);

    mat->d_ind_t = &mat->d_ind[mat->nnz];
    mat->d_ptr_t = &mat->d_ptr[mat->m + 1];
    mat->d_val_t = &mat->d_val[mat->nnz];

    cudaMemcpy(mat->d_ind_t, ind, sizeof(int) * mat->nnz, cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_ptr_t, ptr, sizeof(int) * (mat->n + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_val_t, val, sizeof(real) * mat->nnz, cudaMemcpyHostToDevice);

    mat->fill_transpose();
    
    return mat;
  }

  // d_result = alpha * K * d_rhs + beta * d_result
  bool mv(real *d_rhs,
          real *d_result,
          bool trans,
          real alpha = 1,
          real beta = 0) const;
  
  int nrows() const { return m; }
  int ncols() const { return n; }

protected:
  void fill_transpose() const;
  
  int m;
  int n;
  int nnz;

  cusparseHandle_t cusp_handle;
  cusparseMatDescr_t descr;

  int *d_ind, *d_ind_t;
  int *d_ptr, *d_ptr_t;
  real *d_val, *d_val_t;
};

template<>
inline void sparse_matrix<float>::fill_transpose() const {
  cusparseScsr2csc(cusp_handle, n, m, nnz,
                   d_val_t, d_ptr_t, d_ind_t,
                   d_val, d_ind, d_ptr,
                   CUSPARSE_ACTION_NUMERIC,
                   CUSPARSE_INDEX_BASE_ZERO);
}

template<>
inline void sparse_matrix<double>::fill_transpose() const {
  cusparseDcsr2csc(cusp_handle, n, m, nnz,
                   d_val_t, d_ptr_t, d_ind_t,
                   d_val, d_ind, d_ptr,
                   CUSPARSE_ACTION_NUMERIC,
                   CUSPARSE_INDEX_BASE_ZERO);
}

template<>
inline bool sparse_matrix<float>::mv(float *d_x,
                                     float *d_y,
                                     bool trans,
                                     float alpha,
                                     float beta) const {
  cusparseStatus_t stat;
  
  if(trans)
    stat = cusparseScsrmv(cusp_handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n,
                          m,
                          nnz,
                          &alpha,
                          descr,
                          d_val_t,
                          d_ptr_t,
                          d_ind_t,
                          d_x,
                          &beta,
                          d_y);
  else
    stat = cusparseScsrmv(cusp_handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          m,
                          n,
                          nnz,
                          &alpha,
                          descr,
                          d_val,
                          d_ptr,
                          d_ind,
                          d_x,
                          &beta,
                          d_y);

  return (stat == CUSPARSE_STATUS_SUCCESS);
}

template<>
inline bool sparse_matrix<double>::mv(double *d_x,
                                      double *d_y,
                                      bool trans,
                                      double alpha,
                                      double beta) const {
  cusparseStatus_t stat;
  
  if(trans)
    stat = cusparseDcsrmv(cusp_handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n,
                          m,
                          nnz,
                          &alpha,
                          descr,
                          d_val_t,
                          d_ptr_t,
                          d_ind_t,
                          d_x,
                          &beta,
                          d_y);
  else
    stat = cusparseDcsrmv(cusp_handle,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          m,
                          n,
                          nnz,
                          &alpha,
                          descr,
                          d_val,
                          d_ptr,
                          d_ind,
                          d_x,
                          &beta,
                          d_y);

  return (stat == CUSPARSE_STATUS_SUCCESS);
}

template<typename real>
real normest(const sparse_matrix<real>& A, real tol = 1e-6, int max_iter = 100)
{
  cublasHandle_t handle;
  cublasCreate(&handle);

  int n = A.ncols();
  int m = A.nrows();
  
  real *x, *Ax, *h_x;
  cudaMalloc((void **)&x, sizeof(real) * n);
  cudaMalloc((void **)&Ax, sizeof(real) * m);

  h_x = new real[n];
  for(int i = 0; i < n; ++i)
    h_x[i] = (real) (rand()) / (real)(RAND_MAX);
  cudaMemcpy(x, h_x, sizeof(real) * n, cudaMemcpyHostToDevice);
  
  real norm = 0, norm_prev;

  for(int i = 0; i < max_iter; i++)
  {
    norm_prev = norm;
    A.mv(x, Ax, false, 1, 0);
    cudaDeviceSynchronize();
    A.mv(Ax, x, true, 1, 0); 
    cudaDeviceSynchronize();
    real nx = cuwrapper::nrm2<real>(handle, x, n);
    real nAx = cuwrapper::nrm2<real>(handle, Ax, m);
    cuwrapper::scal<real>(handle, x, real(1) / nx, n);
    norm = nx / nAx;

    if(std::abs(norm_prev - norm) < tol * norm)
      break;
  }

  delete [] h_x;
  
  cudaFree(x);
  cudaFree(Ax);
  cublasDestroy(handle);

  return norm;
}

#endif
