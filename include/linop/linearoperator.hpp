#ifndef LINEAROPERATOR_HPP_
#define LINEAROPERATOR_HPP_

#include <cstdlib>
#include <memory>
#include <vector>

#include <thrust/device_vector.h>

#include "block.hpp"

///
/// \brief Linear operator built out of blocks.
/// 
template<typename T>
class LinearOperator 
{
public:
  LinearOperator();
  virtual ~LinearOperator();

  void AddBlock(std::shared_ptr<Block<T> > block);
  
  void Initialize();
  void Release();

  void Eval(
    thrust::device_vector<T>& result, 
    const thrust::device_vector<T>& rhs);

  void EvalAdjoint(
    thrust::device_vector<T>& result, 
    const thrust::device_vector<T>& rhs);

  /// \brief For debugging/testing purposes. 
  void Eval(
    std::vector<T>& result,
    const std::vector<T>& rhs);

  /// \brief For debugging/testing purposes. 
  void EvalAdjoint(
    std::vector<T>& result,
    const std::vector<T>& rhs);

  /// \brief Returns \sum_{col=1}^{ncols} |K_{row,col}|^{\alpha}.
  T row_sum(size_t row, T alpha) const;

  /// \brief Returns \sum_{row=1}^{nrows} |K_{row,col}|^{\alpha}.
  T col_sum(size_t col, T alpha) const;

  /// \brief Estimates the norm of the linear operator via power iteration.
  T normest(T tol = 1e-6, int max_iters = 100);

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; } 

  size_t gpu_mem_amount() const;
  
protected:
  std::vector<std::shared_ptr<Block<T> > > blocks_;
  size_t nrows_;
  size_t ncols_;

private:
  bool RectangleOverlap(
    size_t x1, size_t y1,
    size_t x2, size_t y2,
    size_t a1, size_t b1,
    size_t a2, size_t b2);
};

#endif
