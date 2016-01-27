#include "linop/block_sparse.hpp"

using namespace linop;

template<typename T>
BlockSparse<T>::BlockSparse(size_t row,
                            size_t col,
                            std::unique_ptr<SparseMatrix<T>> mat)
    : Block<T>(row, col, mat->nrows(), mat->ncols()), mat_(std::move(mat)) {
  
}
  
template<typename T>
void BlockSparse<T>::EvalLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end) {
  if(!mat_->MultVec(thrust::raw_pointer_cast(&(*rhs_begin)),
                thrust::raw_pointer_cast(&(*res_begin)),
                false,
                static_cast<T>(1),
                static_cast<T>(1)))
      throw PDSolverException("Sparse matrix vector multiplication failed");
}

template<typename T>
void BlockSparse<T>::EvalAdjointLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end) {
  if(!mat_->MultVec(thrust::raw_pointer_cast(&(*rhs_begin)),
                thrust::raw_pointer_cast(&(*res_begin)),
                true,
                static_cast<T>(1),
                static_cast<T>(1)))
      throw PDSolverException("Sparse matrix vector multiplication failed");
}

template<typename T>
T BlockSparse<T>::row_sum(size_t row, T alpha) const {
  return mat_->row_sum(row, alpha);
}

template<typename T>
T BlockSparse<T>::col_sum(size_t col, T alpha) const {
  return mat_->col_sum(col, alpha);
}

template<typename T>
size_t BlockSparse<T>::gpu_mem_amount() const {
    return mat_->gpu_mem_amount();
}


template class BlockSparse<float>;
template class BlockSparse<double>;

