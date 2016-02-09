#ifndef PROST_LINEAROPERATOR_HPP_
#define PROST_LINEAROPERATOR_HPP_

#include <thrust/device_vector.h>

#include "prost/linop/block.hpp"

namespace prost {

///
/// \brief Linear operator built out of blocks.
/// 
template<typename T>
class LinearOperator {
public:
  LinearOperator();
  virtual ~LinearOperator();

  void AddBlock(shared_ptr<Block<T>> block);
  
  void Initialize();
  void Release();

  void Eval(
    device_vector<T>& result, 
    const device_vector<T>& rhs);

  void EvalAdjoint(
    device_vector<T>& result, 
    const device_vector<T>& rhs);

  /// \brief For debugging/testing purposes. 
  void Eval(
    vector<T>& result,
    const vector<T>& rhs);

  /// \brief For debugging/testing purposes. 
  void EvalAdjoint(
    vector<T>& result,
    const vector<T>& rhs);

  /// \brief Returns \sum_{col=1}^{ncols} |K_{row,col}|^{\alpha}.
  T row_sum(size_t row, T alpha) const;

  /// \brief Returns \sum_{row=1}^{nrows} |K_{row,col}|^{\alpha}.
  T col_sum(size_t col, T alpha) const;

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; } 

  size_t gpu_mem_amount() const;
  
protected:
  vector<shared_ptr<Block<T>>> blocks_;
  size_t nrows_;
  size_t ncols_;

private:
  bool RectangleOverlap(
    size_t x1, size_t y1,
    size_t x2, size_t y2,
    size_t a1, size_t b1,
    size_t a2, size_t b2);
};

} // namespace prost

#endif // PROST_LINEAROPERATOR_HPP_
