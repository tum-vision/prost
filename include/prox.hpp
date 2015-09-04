#ifndef PROX_HPP_
#define PROX_HPP_

#include "config.hpp"

/**
 * @brief Virtual base class for all proximal operators. Implements prox
 *        for sum of separable functions:
 *
 *        sum_{i=index_}^{index_+count_} f_i(x_i),
 *
 *        where the f_i and x_i are dim_ dimensional.
 */
class Prox {
public:
  Prox(int index, int count, int dim, bool interleaved, bool diagsteps)
      : index_(index),
        count_(count),
        dim_(dim),
        interleaved_(interleaved),
        diagsteps_(diagsteps) {}

  Prox(const Prox& other)
      : index_(other.index_),
        count_(other.count_),
        dim_(other.dim_),
        interleaved_(other.interleaved_),
        diagsteps_(other.diagsteps_) {}
  
  virtual ~Prox() {}

  /**
   * @brief Evaluates the prox operator on the GPU.
   * @param Input argument.
   * @param Result of prox.
   * @param Scalar step size.
   * @param If not set to NULL, specifies diagonal matirx step sizes.
   * @param Perform the prox with inverted step sizes.
   */
  virtual void Evaluate(
      real *d_arg,
      real *d_result,
      real tau,
      real *d_tau,
      bool invert_step = false) = 0;

  // returns amount of gpu memory required in bytes
  virtual int gpu_mem_amount() { return 0; }
  
  int index() const { return index_; }
  int dim() const { return dim_; }
  int count() const { return count_; }
  bool interleaved() const { return interleaved_; }
  bool diagsteps() const { return diagsteps_; }
  int end() const { return index_ + count_ * dim_ - 1; }
  
protected:
  int index_; 
  int dim_;
  int count_; 
  bool interleaved_; // ordering of elements if dim > 1
  bool diagsteps_; // able to handle diagonal matrices as step size?
};

#endif
