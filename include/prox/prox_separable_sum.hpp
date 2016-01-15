#ifndef PROX_SEPARABLE_SUM_HPP_
#define PROX_SEPARABLE_SUM_HPP_

#include <stddef.h>

#include "prox.hpp"

/**
 * @brief Virtual base class for all proximal operators. Implements prox
 *        for sum of separable functions:
 *
 *        sum_{i=index_}^{index_+count_} f_i(x_i),
 *
 *        where the f_i and x_i are dim_ dimensional.
 *
 *        interleaved_ describes the ordering of the elements if dim_ > 1.
 *        If it is set to true, then successive elements in x correspond
 *        to one of count_ many dim_-dimensional vectors.
 *        If interleaved_ is set of false, then there are dim_ contigiuous
 *        chunks of count_ many elements.
 *
 */
template<typename T, class ELEM_OPERATION>
class ProxSeparableSum : public Prox<T> {
public:
 class Vector {
 public:
  Vector(ProxSeparableSum<T, ELEM_OPERATION>* parent, T* data, tx) : 
      parent_(parent),
      data_(data),
      tx_(tx) {

  }

  inline __device__ T operator[](size_t i) {
    // Out of bounds check?

    index = parent_.interleaved_ ? (tx * ELEM_OPERATION::dim + i) : (tx + parent_.count * i);

    return data[index];
  }
  private:
    size_t tx_;

    T* data_;
    ProxSeparableSum<T, ELEM_OPERATION>* parent_;
  }; 
    
  ProxSeparableSum(size_t index, size_t count);

  
  virtual ~Prox() {}

  /**
   * @brief Initializes the prox Operator, copies data to the GPU.
   *
   */
  virtual bool Init();

  /**
   * @brief Cleans up GPU data.
   *
   */
  virtual void Release() {}

  /**
   * @brief Evaluates the prox operator on the GPU. d_arg, d_result and
   *        d_tau are all pointers pointing to the whole vector in memory.
   *
   * @param Proximal operator argument.
   * @param Result of prox.
   * @param Scalar step size.
   * @param Diagonal step sizes.
   */
  void Eval(T *d_arg, T *d_res, T *d_tau, T tau);

  // set/get methods
  virtual size_t gpu_mem_amount() { return 0; }  
  
protected:
  /**
   * @brief Evaluates the prox operator on the GPU, local meaning that
   *        d_arg, d_res and d_tau point to the place in memory where the
   *        prox begins.
   *
   * @param Proximal operator argument.
   * @param Result of prox.
   * @param Scalar step size.
   * @param Diagonal step sizes.
   * @param Perform the prox with inverted step sizes?
   *
   */
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
  
  
};

#endif
