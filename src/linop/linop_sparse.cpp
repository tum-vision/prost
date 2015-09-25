#include "linop_sparse.hpp"

template<typename T>
LinOpSparse<T>::LinOpSparse(size_t row,
                            size_t col,
                            SparseMatrix<T> *mat)
    : LinOp<T>(row, col, mat->nrows(), mat->ncols()), mat_(mat)
{
  
}

template<typename T>
LinOpSparse<T>::~LinOpSparse() {
  Release();
}

template<typename T>
bool LinOpSparse<T>::Init() {
  return true;
}

template<typename T>
void LinOpSparse<T>::Release() {
  delete mat_;
}
  
template<typename T>
void LinOpSparse<T>::EvalLocalAdd(T *d_res, T *d_rhs) {
  mat_->MultVec(d_rhs,
                d_res,
                false,
                static_cast<T>(1),
                static_cast<T>(1));
}

template<typename T>
void LinOpSparse<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {
  mat_->MultVec(d_rhs,
                d_res,
                true,
                static_cast<T>(1),
                static_cast<T>(1));
}

template<typename T>
T LinOpSparse<T>::row_sum(int row, T alpha) const {
  return mat_->row_sum(row - row_, alpha);
}

template<typename T>
T LinOpSparse<T>::col_sum(int col, T alpha) const {
  return mat_->col_sum(col - col_, alpha);
}

