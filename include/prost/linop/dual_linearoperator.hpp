#ifndef PROST_DUAL_LINEAROPERATOR_HPP_
#define PROST_DUAL_LINEAROPERATOR_HPP_

#include "prost/linop/linearoperator.hpp"

namespace prost {

/// 
/// \brief Evaluates the negative transpose of the child linear operator.
/// 
template<typename T>
class DualLinearOperator : public LinearOperator<T> {
 public:
  DualLinearOperator(shared_ptr<LinearOperator<T>> child);
  virtual ~DualLinearOperator();

  virtual void Eval(
    device_vector<T>& result, 
    const device_vector<T>& rhs);

  virtual void EvalAdjoint(
    device_vector<T>& result, 
    const device_vector<T>& rhs);
  
    /// \brief Returns \sum_{col=1}^{ncols} |K_{row,col}|^{\alpha}.
  virtual T row_sum(size_t row, T alpha) const;

  /// \brief Returns \sum_{row=1}^{nrows} |K_{row,col}|^{\alpha}.
  virtual T col_sum(size_t col, T alpha) const;

  virtual size_t nrows() const;
  virtual size_t ncols() const;

  virtual size_t gpu_mem_amount() const { return child_->gpu_mem_amount(); }

 protected:
  shared_ptr<LinearOperator<T>> child_;
};

}

#endif // PROST_DUAL_LINEAROPERATOR_HPP_
