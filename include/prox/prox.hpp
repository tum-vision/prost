#ifndef PROX_HPP_
#define PROX_HPP_

#include <stddef.h>

template<typename T> class ProxMoreau;
template<typename T> class ProxPlusLinterm;

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
template<typename T>
class Prox {
  friend class ProxMoreau<T>;
  friend class ProxPlusLinterm<T>;
  
public:
  Prox(size_t index, size_t count, size_t dim, bool interleaved, bool diagsteps) :
    index_(index),
    count_(count),
    dim_(dim),
    interleaved_(interleaved),
    diagsteps_(diagsteps) { }

  Prox(const Prox<T>& other) :
    index_(other.index_),
    count_(other.count_),
    dim_(other.dim_),
    interleaved_(other.interleaved_),
    diagsteps_(other.diagsteps_) { }
  
  virtual ~Prox() {}

  /**
   * @brief Initializes the prox Operator, copies data to the GPU.
   *
   */
  virtual bool Init() { return true; }

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
  size_t index() const { return index_; }
  size_t dim() const { return dim_; }
  size_t count() const { return count_; }
  bool interleaved() const { return interleaved_; }
  bool diagsteps() const { return diagsteps_; }
  size_t end() const { return index_ + count_ * dim_ - 1; }
  
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
                         bool invert_tau) = 0;
  
  size_t index_; 
  size_t count_; 
  size_t dim_;
  bool interleaved_; // ordering of elements if dim > 1
  bool diagsteps_; // able to handle diagonal matrices as step size?
};

#endif
