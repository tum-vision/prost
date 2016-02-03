#ifndef PROX_SEPARABLE_SUM_HPP_
#define PROX_SEPARABLE_SUM_HPP_

#include "prox/prox.hpp"

namespace prox
{

/// 
/// \brief Abstract base class for proximal operators 
///        for a sum of separable functions:
/// 
///        sum_{i=index_}^{index_+count_} f_i(x_i),
/// 
///        where the f_i and x_i are dim_ dimensional.
/// 
///        interleaved_ describes the ordering of the elements if dim_ > 1.
///        If it is set to true, then successive elements in x correspond
///        to one of count_ many dim_-dimensional vectors.
///        If interleaved_ is set of false, then there are dim_ contigiuous
///        chunks of count_ many elements.
/// 
template<typename T>
class ProxSeparableSum : public Prox<T> 
{
public:
  ProxSeparableSum(size_t index, 
                   size_t count, 
                   size_t dim, 
                   bool interleaved,
                   bool diagsteps)
      
      : Prox<T>(index, count*dim, diagsteps),
      count_(count),
      dim_(dim),
      interleaved_(interleaved) { }

  size_t dim() const { return dim_; }
  size_t count() const { return count_; }
  bool interleaved() const { return interleaved_; }

protected:
  size_t count_; 
  size_t dim_;

  /// \brief Ordering of elements if dim > 1.
  bool interleaved_; 
};

}

#endif
