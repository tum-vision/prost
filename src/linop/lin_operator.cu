#include "linop/lin_operator.hpp"

#include <iostream>
#include <cuda_runtime.h>

using namespace linop;

template<typename T>
LinOperator<T>::LinOperator() {
  nrows_ = 0;
  ncols_ = 0;
}

template<typename T>
LinOperator<T>::~LinOperator() {
  Release();
}

// careful: transfers ownership to LinOperator
template<typename T>
void LinOperator<T>::AddBlock(std::unique_ptr<Block<T>>op) {
  blocks_.push_back(std::move(op));
}

template<typename T>
void LinOperator<T>::Init() {
  // check if any two linear operators overlap
  nrows_ = 0;
  ncols_ = 0;

  size_t area = 0;
  
  bool overlap = false;
  for(size_t i = 0; i < blocks_.size(); i++) {
    Block<T>* op_i = blocks_[i].get();

    nrows_ = std::max(op_i->row() + op_i->nrows(), nrows_);
    ncols_ = std::max(op_i->col() + op_i->ncols(), ncols_);

    area += op_i->nrows() * op_i->ncols();
    
    for(size_t j = i + 1; j < blocks_.size(); j++) {
      Block<T>* op_j = blocks_[j].get();

      overlap |= RectangleOverlap(op_i->col(), op_i->row(),
                                  op_i->col() + op_i->ncols() - 1,
                                  op_i->row() + op_i->nrows() - 1,
                                  op_j->col(), op_j->row(),
                                  op_j->col() + op_j->ncols() - 1,
                                  op_j->row() + op_j->nrows() - 1);      
    }
  }

  if(overlap) {
    throw PDSolverException("linop::Blocks in LinOperator are overlapping!");
  }

  // this should even work with overflows due to modular arithmetic :-)
  if(area != nrows_ * ncols_)  {
    throw PDSolverException("There's empty space between the linop::Blocks!");
  }

  for(auto& b : blocks_)
    b->Init();

}

template<typename T>
void LinOperator<T>::Release() {
}

template<typename T>
void LinOperator<T>::Eval(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs) {
  thrust::fill(res.begin(), res.end(), 0);

  for(auto& b : blocks_)
    b->EvalAdd(res, rhs);
}

template<typename T>
void LinOperator<T>::EvalAdjoint(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs) {
  thrust::fill(res.begin(), res.end(), 0);

  for(auto& b : blocks_)
    b->EvalAdjointAdd(res, rhs);
}

template<typename T>
T LinOperator<T>::row_sum(size_t row, T alpha) const {
  T sum = 0;
  
  for(auto& b : blocks_) {
    if(row < b->row() ||
       row >= (b->row() + b->nrows()))
      continue;
    
    sum += b->row_sum(row - b->row(), alpha);
  }

  return sum;
}

template<typename T>
T LinOperator<T>::col_sum(size_t col, T alpha) const {
  T sum = 0;
  
  for(auto& b : blocks_) {
    if(col < b->col() ||
       col >= (b->col() + b->ncols()))
      continue;
    
    sum += b->col_sum(col - b->col(), alpha);
  }

  return sum;
}

template<typename T>
size_t LinOperator<T>::gpu_mem_amount() const {
  size_t mem_amount = 0;

  for(auto& b : blocks_) 
    mem_amount += b->gpu_mem_amount();

  return mem_amount;
}

// Explicit template instantiation
template class LinOperator<float>;
template class LinOperator<double>;
