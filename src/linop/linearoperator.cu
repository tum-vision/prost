#include <algorithm>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/fill.h>

#include "prost/linop/linearoperator.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
bool LinearOperator<T>::RectangleOverlap(
  size_t x1, size_t y1,
  size_t x2, size_t y2,
  size_t a1, size_t b1,
  size_t a2, size_t b2) 
{ 
  return (x1 <= a2) && (x2 >= a1) && (y1 <= b2) && (y2 >= b1);
}

// used for sorting blocks
template<typename T>
struct BlockCompareRow {
  bool operator()(Block<T>* const& left, Block<T>* const& right) {
    if(left->row() < right->row())
      return true;

    return false;
  }
};

template<typename T>
struct BlockCompareCol 
{
  bool operator()(Block<T>* const& left, Block<T>* const& right) 
  {
    if(left->col() < right->col())
      return true;

    return false;
  }
};

template<typename T>
LinearOperator<T>::LinearOperator() 
{
  nrows_ = 0;
  ncols_ = 0;
}

template<typename T>
LinearOperator<T>::~LinearOperator() 
{
  Release();
}

template<typename T>
void LinearOperator<T>::AddBlock(std::shared_ptr<Block<T>> block) 
{
  blocks_.push_back(block);
}

template<typename T>
void LinearOperator<T>::Initialize() 
{
  // check if any two linear operators overlap
  nrows_ = 0;
  ncols_ = 0;

  size_t area = 0;
  
  bool overlap = false;
  for(size_t i = 0; i < blocks_.size(); i++) 
  {
    std::shared_ptr<Block<T> > block_i = blocks_[i];

    nrows_ = std::max(block_i->row() + block_i->nrows(), nrows_);
    ncols_ = std::max(block_i->col() + block_i->ncols(), ncols_);

    area += block_i->nrows() * block_i->ncols();
    
    for(size_t j = i + 1; j < blocks_.size(); j++) 
    {
      std::shared_ptr<Block<T> > block_j = blocks_[j];

      overlap |= RectangleOverlap(block_i->col(), block_i->row(),
                                  block_i->col() + block_i->ncols() - 1,
                                  block_i->row() + block_i->nrows() - 1,
                                  block_j->col(), block_j->row(),
                                  block_j->col() + block_j->ncols() - 1,
                                  block_j->row() + block_j->nrows() - 1);      
    }
  }

  if(overlap) 
    throw Exception("Blocks are overlapping inside the linear operator. Recheck the indices.");

/*
  if(area != nrows_ * ncols_)  
    std::cout << "Warning: There's empty space between the blocks inside the linear operator. Recheck the indicies." << std::endl;
*/

  for(auto& block : blocks_)
    block->Initialize();
}

template<typename T>
void LinearOperator<T>::Release()
{
  for(auto& block : blocks_)
    block->Release();
}

template<typename T>
void LinearOperator<T>::Eval(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& rhs)
{
  thrust::fill(result.begin(), result.end(), 0);

  for(auto& block : blocks_)
    block->EvalAdd(result, rhs);
}

template<typename T>
void LinearOperator<T>::EvalAdjoint(
  thrust::device_vector<T>& result, 
  const thrust::device_vector<T>& rhs)
{
  thrust::fill(result.begin(), result.end(), 0);

  for(auto& block : blocks_)
    block->EvalAdjointAdd(result, rhs);
}

template<typename T>
double LinearOperator<T>::Eval(
  std::vector<T>& result,
  const std::vector<T>& rhs)
{
  static const int repeats = 5;
  
  thrust::device_vector<T> d_rhs(rhs.begin(), rhs.end());
  thrust::device_vector<T> d_res;
  d_res.resize(nrows());

  const clock_t begin_time = clock();
  for(int i = 0; i < repeats; i++)
  {
    Eval(d_res, d_rhs);
    cudaDeviceSynchronize();
  }
  double s = (double)(clock() - begin_time) / CLOCKS_PER_SEC;

  result.resize(nrows());
  thrust::copy(d_res.begin(), d_res.end(), result.begin());

  return (s * 1000 / (double)repeats);
}

template<typename T>
double LinearOperator<T>::EvalAdjoint(
  std::vector<T>& result,
  const std::vector<T>& rhs)
{
  static const int repeats = 5;

  thrust::device_vector<T> d_rhs(rhs.begin(), rhs.end());
  thrust::device_vector<T> d_res;
  d_res.resize(ncols());

  const clock_t begin_time = clock();
  for(int i = 0; i < repeats; i++)
  {
    EvalAdjoint(d_res, d_rhs);
    cudaDeviceSynchronize();
  }
  double s = (double)(clock() - begin_time) / CLOCKS_PER_SEC;

  result.resize(ncols());
  thrust::copy(d_res.begin(), d_res.end(), result.begin());

  return (s * 1000 / (double)repeats);
}

template<typename T>
T LinearOperator<T>::row_sum(size_t row, T alpha) const 
{
  T sum = 0;
  
  for(auto& block : blocks_)
  {
    if(row < block->row() ||
       row >= (block->row() + block->nrows()))
      continue;
    
    sum += block->row_sum(row - block->row(), alpha);
  }

  return sum;
}

template<typename T>
T LinearOperator<T>::col_sum(size_t col, T alpha) const 
{
  T sum = 0;
  for(auto& block : blocks_) 
  {
    if(col < block->col() ||
       col >= (block->col() + block->ncols()))
      continue;
    
    sum += block->col_sum(col - block->col(), alpha);
  }

  return sum;
}

template<typename T>
size_t LinearOperator<T>::gpu_mem_amount() const 
{
  size_t mem = 0;

  for(auto& block : blocks_)
    mem += block->gpu_mem_amount();

  return mem;
}

// Explicit template instantiation
template class LinearOperator<float>;
template class LinearOperator<double>;

} // namespace prost