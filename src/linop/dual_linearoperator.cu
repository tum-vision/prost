#include "prost/linop/dual_linearoperator.hpp"

namespace prost {

template<typename T>
DualLinearOperator<T>::DualLinearOperator(shared_ptr<LinearOperator<T>> child)
    : child_(child)
{
  this->nrows_ = child_->ncols();
  this->ncols_ = child_->nrows();
}

template<typename T>
DualLinearOperator<T>::~DualLinearOperator()
{
}

template<typename T>
void DualLinearOperator<T>::Eval(
    device_vector<T>& result, 
    const device_vector<T>& rhs)
{
  thrust::fill(result.begin(), result.end(), 0);

  for(auto& block : child_->blocks_)
    block->EvalAdjointAdd(result, rhs);

  thrust::transform(result.begin(), result.end(), result.begin(),
                    thrust::negate<float>());
}

template<typename T>
void DualLinearOperator<T>::EvalAdjoint(
    device_vector<T>& result, 
    const device_vector<T>& rhs)
{
  thrust::fill(result.begin(), result.end(), 0);

  for(auto& block : child_->blocks_)
    block->EvalAdd(result, rhs);

  thrust::transform(result.begin(), result.end(), result.begin(),
                    thrust::negate<float>());
}
  
template<typename T>
T DualLinearOperator<T>::row_sum(size_t row, T alpha) const
{
  return child_->col_sum(row, alpha);
}

template<typename T>
T DualLinearOperator<T>::col_sum(size_t col, T alpha) const
{
  return child_->row_sum(col, alpha);
}

template<typename T>
size_t DualLinearOperator<T>::nrows() const
{
  return child_->ncols();
}

template<typename T>
size_t DualLinearOperator<T>::ncols() const
{
  return child_->nrows();
}

// Explicit template instantiation
template class DualLinearOperator<float>;
template class DualLinearOperator<double>;

}