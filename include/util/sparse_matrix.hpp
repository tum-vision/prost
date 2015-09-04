#ifndef SPARSE_MATRIX_HPP_
#define SPARSE_MATRIX_HPP_

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cmath>
#include <cstdlib>

#include "util/cuwrap.hpp"

/**
 * @brief Wrapper class around cuSPARSE API.
 */
template<typename real>
class SparseMatrix {
  SparseMatrix() { }
  
public:
  virtual ~SparseMatrix() {
    cudaFree(d_ind_);
    cudaFree(d_val_);
    cudaFree(d_ptr_);

    delete [] h_ind_;
    delete [] h_val_;
    delete [] h_ptr_;

    delete [] h_ind_t_;
    delete [] h_val_t_;
    delete [] h_ptr_t_;
  }

  static SparseMatrix<real> *CreateFromCSC(
      int m,
      int n,
      int nnz,
      real *val,
      int *ptr,
      int *ind)
  {
    SparseMatrix<real> *mat = new SparseMatrix<real>;

    cusparseCreate(&mat->cusp_handle_);
    cusparseCreateMatDescr(&mat->descr_);
    cusparseSetMatType(mat->descr_, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(mat->descr_, CUSPARSE_INDEX_BASE_ZERO);

    mat->m_ = m;
    mat->n_ = n;
    mat->nnz_ = nnz;

    cudaMalloc((void **)&mat->d_ind_, sizeof(int) * 2 * mat->nnz_);
    cudaMalloc((void **)&mat->d_ptr_, sizeof(int) * (mat->m_ + mat->n_ + 2));
    cudaMalloc((void **)&mat->d_val_, sizeof(real) * 2 * mat->nnz_);

    mat->d_ind_t_ = &mat->d_ind_[mat->nnz_];
    mat->d_ptr_t_ = &mat->d_ptr_[mat->m_ + 1];
    mat->d_val_t_ = &mat->d_val_[mat->nnz_];

    cudaMemcpy(mat->d_ind_t_, ind, sizeof(int) * mat->nnz_, cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_ptr_t_, ptr, sizeof(int) * (mat->n_ + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(mat->d_val_t_, val, sizeof(real) * mat->nnz_, cudaMemcpyHostToDevice);

    mat->FillTranspose();

    mat->h_ind_ = new int[mat->nnz_];
    mat->h_ptr_ = new int[mat->m_ + 1];
    mat->h_val_ = new real[mat->nnz_];

    mat->h_ind_t_ = new int[mat->nnz_];
    mat->h_ptr_t_ = new int[mat->n_ + 1];
    mat->h_val_t_ = new real[mat->nnz_];

    cudaMemcpy(mat->h_ind_, mat->d_ind_, sizeof(int) * mat->nnz_, cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->h_ptr_, mat->d_ptr_, sizeof(int) * (mat->m_ + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->h_val_, mat->d_val_, sizeof(int) * mat->nnz_, cudaMemcpyDeviceToHost);

    cudaMemcpy(mat->h_ind_t_, mat->d_ind_t_, sizeof(int) * mat->nnz_, cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->h_ptr_t_, mat->d_ptr_t_, sizeof(int) * (mat->n_ + 1), cudaMemcpyDeviceToHost);
    cudaMemcpy(mat->h_val_t_, mat->d_val_t_, sizeof(int) * mat->nnz_, cudaMemcpyDeviceToHost);
    
    return mat;
  }

  // d_result = alpha * K * d_rhs + beta * d_result
  bool MultVec(
      real *d_rhs,
      real *d_result,
      bool trans,
      real alpha = 1,
      real beta = 0) const;
  
  int nrows() const { return m_; }
  int ncols() const { return n_; }

  real row_sum(int row, real alpha) const {
    real sum = 0;
    for(int i = h_ptr_[row]; i < h_ptr_[row + 1]; i++)
      sum += pow(abs(h_val_[i]), alpha);
    
    return sum;
  }

  real col_sum(int col, real alpha) const {
    real sum = 0;
    for(int i = h_ptr_t_[col]; i < h_ptr_t_[col + 1]; i++)
      sum += pow(abs(h_val_t_[i]), alpha);
    
    return sum;
  }

  int gpu_mem_amount() const {
    int total_bytes = 0;

    total_bytes += 2 * nnz_ * sizeof(int);
    total_bytes += (m_ + n_ + 2) * sizeof(int);
    total_bytes += 2 * nnz_ * sizeof(real);

    return total_bytes;
  }

protected:
  void FillTranspose() const;
  
  int m_; // number of rows
  int n_; // number of cols
  int nnz_; // number of non zero elements

  cusparseHandle_t cusp_handle_;
  cusparseMatDescr_t descr_;

  int *d_ind_, *d_ind_t_;
  int *d_ptr_, *d_ptr_t_;
  real *d_val_, *d_val_t_;

  int *h_ind_, *h_ind_t_;
  int *h_ptr_, *h_ptr_t_;
  real *h_val_, *h_val_t_;
};

template<>
inline void SparseMatrix<float>::FillTranspose() const {
  cusparseScsr2csc(cusp_handle_, n_, m_, nnz_,
                   d_val_t_, d_ptr_t_, d_ind_t_,
                   d_val_, d_ind_, d_ptr_,
                   CUSPARSE_ACTION_NUMERIC,
                   CUSPARSE_INDEX_BASE_ZERO);
}

template<>
inline void SparseMatrix<double>::FillTranspose() const {
  cusparseDcsr2csc(cusp_handle_, n_, m_, nnz_,
                   d_val_t_, d_ptr_t_, d_ind_t_,
                   d_val_, d_ind_, d_ptr_,
                   CUSPARSE_ACTION_NUMERIC,
                   CUSPARSE_INDEX_BASE_ZERO);
}

template<>
inline bool SparseMatrix<float>::MultVec(
    float *d_x,
    float *d_y,
    bool trans,
    float alpha,
    float beta) const
{
  cusparseStatus_t stat;
  
  if(trans)
    stat = cusparseScsrmv(cusp_handle_,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n_,
                          m_,
                          nnz_,
                          &alpha,
                          descr_,
                          d_val_t_,
                          d_ptr_t_,
                          d_ind_t_,
                          d_x,
                          &beta,
                          d_y);
  else
    stat = cusparseScsrmv(cusp_handle_,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          m_,
                          n_,
                          nnz_,
                          &alpha,
                          descr_,
                          d_val_,
                          d_ptr_,
                          d_ind_,
                          d_x,
                          &beta,
                          d_y);

  return (stat == CUSPARSE_STATUS_SUCCESS);
}

template<>
inline bool SparseMatrix<double>::MultVec(
    double *d_x,
    double *d_y,
    bool trans,
    double alpha,
    double beta) const
{
  cusparseStatus_t stat;
  
  if(trans)
    stat = cusparseDcsrmv(cusp_handle_,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          n_,
                          m_,
                          nnz_,
                          &alpha,
                          descr_,
                          d_val_t_,
                          d_ptr_t_,
                          d_ind_t_,
                          d_x,
                          &beta,
                          d_y);
  else
    stat = cusparseDcsrmv(cusp_handle_,
                          CUSPARSE_OPERATION_NON_TRANSPOSE,
                          m_,
                          n_,
                          nnz_,
                          &alpha,
                          descr_,
                          d_val_,
                          d_ptr_,
                          d_ind_,
                          d_x,
                          &beta,
                          d_y);

  return (stat == CUSPARSE_STATUS_SUCCESS);
}

// TODO: understand how this works. power iteration?
template<typename real>
real MatrixNormest(const SparseMatrix<real>& A, real tol = 1e-6, int max_iter = 100)
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
    
    A.MultVec(x, Ax, false, 1, 0);
    A.MultVec(Ax, x, true, 1, 0); 
    
    real nx = cuwrap::nrm2<real>(handle, x, n);
    real nAx = cuwrap::nrm2<real>(handle, Ax, m);
    cuwrap::scal<real>(handle, x, real(1) / nx, n);
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
