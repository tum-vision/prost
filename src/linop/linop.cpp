#include "linop/linop.hpp"

#include <iostream>
#include <cuda_runtime.h>

bool RectangleOverlap(size_t x1, size_t y1,
                      size_t x2, size_t y2,
                      size_t a1, size_t b1,
                      size_t a2, size_t b2) 
{ 
  return (x1 <= a2) && (x2 >= a1) && (y1 <= b2) && (y2 >= b1);
}

// used for sorting linear operators
template<typename T>
struct LinOpCompareRow {
  bool operator()(LinOp<T>* const& left, LinOp<T>* const& right) {
    if(left->row() < right->row())
      return true;

    return false;
  }
};

template<typename T>
struct LinOpCompareCol {
  bool operator()(LinOp<T>* const& left, LinOp<T>* const& right) {
    if(left->col() < right->col())
      return true;

    return false;
  }
};

template<typename T>
LinOp<T>::LinOp(size_t row, size_t col, size_t nrows, size_t ncols)
    : row_(row), col_(col), nrows_(nrows), ncols_(ncols)
{
}

template<typename T>
LinOp<T>::~LinOp() {
}

template<typename T>
void LinOp<T>::EvalAdd(T *d_res, T *d_rhs) {
  EvalLocalAdd(&d_res[row_], &d_rhs[col_]);
}

template<typename T>
void LinOp<T>::EvalAdjointAdd(T *d_res, T *d_rhs) {
  EvalAdjointLocalAdd(&d_res[col_], &d_rhs[row_]);
}

template<typename T>
void LinOp<T>::EvalLocalAdd(T *d_res, T *d_rhs) {
  // do nothing for zero operator
}

template<typename T>
void LinOp<T>::EvalAdjointLocalAdd(T *d_res, T *d_rhs) {
  // do nothing for zero operator
}

template<typename T>
bool LinOp<T>::Init() {
  return true;
}

template<typename T>
void LinOp<T>::Release() {
}

template<typename T>
LinearOperator<T>::LinearOperator() {
  nrows_ = 0;
  ncols_ = 0;
}

template<typename T>
LinearOperator<T>::~LinearOperator() {
  Release();
}

// careful: transfers ownership to LinearOperator
template<typename T>
void LinearOperator<T>::AddOperator(LinOp<T> *op) {
  operators_.push_back(op);
}

template<typename T>
bool LinearOperator<T>::Init() {
  // check if any two linear operators overlap
  nrows_ = 0;
  ncols_ = 0;

  size_t area = 0;
  
  bool overlap = false;
  for(size_t i = 0; i < operators_.size(); i++) {
    LinOp<T>* op_i = operators_[i];

    nrows_ = std::max(op_i->row() + op_i->nrows(), nrows_);
    ncols_ = std::max(op_i->col() + op_i->ncols(), ncols_);

    area += op_i->nrows() * op_i->ncols();
    
    for(size_t j = i + 1; j < operators_.size(); j++) {
      LinOp<T>* op_j = operators_[j];

      overlap |= RectangleOverlap(op_i->col(), op_i->row(),
                                  op_i->col() + op_i->ncols() - 1,
                                  op_i->row() + op_i->nrows() - 1,
                                  op_j->col(), op_j->row(),
                                  op_j->col() + op_j->ncols() - 1,
                                  op_j->row() + op_j->nrows() - 1);      
    }
  }

  if(overlap)
    return false;

  // this should even work with overflows due to modular arithmetic :-)
  if(area != nrows_ * ncols_) 
    return false;

  for(size_t i = 0; i < operators_.size(); i++)
    if(!operators_[i]->Init())
      return false;

  return true;
}

template<typename T>
void LinearOperator<T>::Release() {
  for(size_t i = 0; i < operators_.size(); i++)
    delete operators_[i];

  operators_.clear();
}

template<typename T>
void LinearOperator<T>::Eval(T *d_res, T *d_rhs) {
  cudaMemset(d_res, 0, sizeof(T) * nrows_);
  
  for(size_t i = 0; i < operators_.size(); i++)
    operators_[i]->EvalAdd(d_res, d_rhs);
}

template<typename T>
void LinearOperator<T>::EvalAdjoint(T *d_res, T *d_rhs) {
  cudaMemset(d_res, 0, sizeof(T) * ncols_);
  
  for(size_t i = 0; i < operators_.size(); i++)
    operators_[i]->EvalAdjointAdd(d_res, d_rhs);
}

template<typename T>
T LinearOperator<T>::row_sum(size_t row, T alpha) const {
  T sum = 0;
  
  for(size_t i = 0; i < operators_.size(); i++) {
    if(row < operators_[i]->row() ||
       row >= (operators_[i]->row() + operators_[i]->nrows()))
      continue;
    
    sum += operators_[i]->row_sum(row - operators_[i]->row(), alpha);
  }

  return sum;
}

template<typename T>
T LinearOperator<T>::col_sum(size_t col, T alpha) const {
  T sum = 0;
  
  for(size_t i = 0; i < operators_.size(); i++) {
    if(col < operators_[i]->col() ||
       col >= (operators_[i]->col() + operators_[i]->ncols()))
      continue;
    
    sum += operators_[i]->col_sum(col - operators_[i]->col(), alpha);
  }

  return sum;
}

template<typename T>
size_t LinearOperator<T>::gpu_mem_amount() const {
  size_t mem = 0;

  for(size_t i = 0; i < operators_.size(); i++) 
    mem += operators_[i]->gpu_mem_amount();

  return mem;
}

// Explicit template instantiation
template class LinOp<float>;
template class LinOp<double>;
template class LinearOperator<float>;
template class LinearOperator<double>;
