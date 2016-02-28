#include "prost/linop/block_sparse_kron_id.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

static const size_t kMaxNNZ = 2048; 
static const size_t kMaxRows = 1024; 
static const size_t kMaxCols = 1024; 

__constant__ int32_t cmem_ind[kMaxNNZ];
__constant__ int32_t cmem_ind_t[kMaxNNZ];
__constant__ float cmem_val[kMaxNNZ];
__constant__ float cmem_val_t[kMaxNNZ];
__constant__ int32_t cmem_ptr[kMaxRows];
__constant__ int32_t cmem_ptr_t[kMaxCols];

template<> size_t BlockSparseKronId<float>::cmem_counter_nnz_ = 0;
template<> size_t BlockSparseKronId<float>::cmem_counter_rows_ = 0;
template<> size_t BlockSparseKronId<float>::cmem_counter_cols_ = 0;
template<> size_t BlockSparseKronId<double>::cmem_counter_nnz_ = 0;
template<> size_t BlockSparseKronId<double>::cmem_counter_rows_ = 0;
template<> size_t BlockSparseKronId<double>::cmem_counter_cols_ = 0;

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
  const int tx = threadIdx.x + blockIdx.x * blockDim.x;

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
__global__ void BlockSparseKronIdKernelConst(
    T *result,
    const T *rhs,
    size_t diaglength,
    size_t nrows,
    size_t idx_rows,
    size_t idx_cols,
    size_t idx_nnz,
    bool transpose)
{
  const int tx = threadIdx.x + blockIdx.x * blockDim.x;

  if(tx >= diaglength * nrows)
    return;
  
  T sum = 0;

  if(transpose)
  {
    size_t col_ofs = tx % diaglength;
    size_t row = tx / diaglength;

    const int32_t stop = cmem_ptr_t[idx_cols + row + 1];
    
    for(int32_t i = cmem_ptr_t[idx_cols + row]; i < stop; i++)
      sum += cmem_val_t[idx_nnz + i] * rhs[cmem_ind_t[idx_nnz + i] * diaglength + col_ofs];

    result[tx] += sum;
  }
  else
  {
    size_t col_ofs = tx % diaglength;
    size_t row = tx / diaglength;

    const int32_t stop = cmem_ptr[idx_rows + row + 1];
    
    for(int32_t i = cmem_ptr[idx_rows + row]; i < stop; i++)
      sum += cmem_val[idx_nnz + i] * rhs[cmem_ind[idx_nnz + i] * diaglength + col_ofs];

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
  BlockSparseKronId<T> *block = new BlockSparseKronId<T>(row, col, m * diaglength, n * diaglength);

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
  cmem_offset_nnz_ = cmem_counter_nnz_;
  cmem_offset_cols_ = cmem_counter_cols_;
  cmem_offset_rows_ = cmem_counter_rows_;

  cmem_counter_nnz_ += mat_nnz_;
  cmem_counter_cols_ += mat_ncols_ + 1;
  cmem_counter_rows_ += mat_nrows_ + 1;

  if((cmem_counter_nnz_ > kMaxNNZ) ||
     (cmem_counter_rows_ > kMaxRows) ||
     (cmem_counter_cols_ > kMaxCols))
  {
    in_cmem_ = false;

    std::cout << cmem_counter_nnz_ << ", " << kMaxNNZ << std::endl;
    std::cout << cmem_counter_rows_ << ", " << kMaxRows << std::endl;
    std::cout << cmem_counter_cols_ << ", " << kMaxCols << std::endl;
  }
  else
  {
    in_cmem_ = true;
  }

  if(in_cmem_)
  {
    cudaMemcpyToSymbol(cmem_ind_t,
                       &host_ind_t_[0],
                       sizeof(int32_t) * mat_nnz_,
                       cmem_offset_nnz_ * sizeof(int32_t));

    cudaMemcpyToSymbol(cmem_ptr_t,
                       &host_ptr_t_[0],
                       sizeof(int32_t) * (mat_ncols_ + 1),
                       cmem_offset_cols_ * sizeof(int32_t));

    cudaMemcpyToSymbol(cmem_val_t,
                       &host_val_t_[0],
                       sizeof(float) * mat_nnz_,
                       cmem_offset_nnz_ * sizeof(float));

    cudaMemcpyToSymbol(cmem_ind,
                       &host_ind_[0],
                       sizeof(int32_t) * mat_nnz_,
                       cmem_offset_nnz_ * sizeof(int32_t));

    cudaMemcpyToSymbol(cmem_ptr,
                       &host_ptr_[0],
                       sizeof(int32_t) * (mat_nrows_ + 1),
                       cmem_offset_rows_ * sizeof(int32_t));

    cudaMemcpyToSymbol(cmem_val,
                       &host_val_[0],
                       sizeof(float) * mat_nnz_,
                       cmem_offset_nnz_ * sizeof(float));
  }
  else
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
  if(in_cmem_)
    return 0;
  else
    return (host_ind_.size() + host_ind_t_.size() + host_ptr_.size() + host_ptr_t_.size()) * sizeof(int32_t) + (host_val_.size() + host_val_t_.size()) * sizeof(T);
}

template<typename T>
void BlockSparseKronId<T>::EvalLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end)
{
  const size_t count = this->nrows();
  const size_t num_blocks = (count + kBlockSizeCUDA) / kBlockSizeCUDA;

  if(!in_cmem_)
  {
    BlockSparseKronIdKernel<T>
        <<<num_blocks, kBlockSizeCUDA>>>(
            thrust::raw_pointer_cast(&(*res_begin)),
            thrust::raw_pointer_cast(&(*rhs_begin)),
            diaglength_,
            mat_nrows_,
            thrust::raw_pointer_cast(ind_.data()),
            thrust::raw_pointer_cast(ptr_.data()),
            thrust::raw_pointer_cast(val_.data()));
  }
  else
  {
    BlockSparseKronIdKernelConst<T>
        <<<num_blocks, kBlockSizeCUDA>>>(
            thrust::raw_pointer_cast(&(*res_begin)),
            thrust::raw_pointer_cast(&(*rhs_begin)),
            diaglength_,
            mat_nrows_,
            cmem_offset_rows_,
            cmem_offset_cols_,
            cmem_offset_nnz_,
            false);

  }
  cudaDeviceSynchronize();
                                                                                                     // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(error) << std::endl;
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
  const size_t count = this->ncols();
  const size_t num_blocks = (count + kBlockSizeCUDA) / kBlockSizeCUDA;

  if(!in_cmem_)
  {
    BlockSparseKronIdKernel<T>
        <<<num_blocks, kBlockSizeCUDA>>>(
            thrust::raw_pointer_cast(&(*res_begin)),
            thrust::raw_pointer_cast(&(*rhs_begin)),
            diaglength_,
            mat_ncols_,
            thrust::raw_pointer_cast(ind_t_.data()),
            thrust::raw_pointer_cast(ptr_t_.data()),
            thrust::raw_pointer_cast(val_t_.data()));
  }
  else
  {
    BlockSparseKronIdKernelConst<T>
        <<<num_blocks, kBlockSizeCUDA>>>(
            thrust::raw_pointer_cast(&(*res_begin)),
            thrust::raw_pointer_cast(&(*rhs_begin)),
            diaglength_,
            mat_ncols_,
            cmem_offset_rows_,
            cmem_offset_cols_,
            cmem_offset_nnz_,
            true);
  }
  cudaDeviceSynchronize();
                                                                                                     // check for error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

// Explicit template instantiation
template class BlockSparseKronId<float>;
template class BlockSparseKronId<double>;

} // namespace prost