#ifndef PROX_HPP_
#define PROX_HPP_

#include <stddef.h>
#include <thrust/device_vector.h>


namespace prox {
template<typename T> class ProxMoreau;

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
  
public:
  Prox(size_t index, size_t size, bool diagsteps) :
    index_(index),
    size_(size),
    diagsteps_(diagsteps) { }

  Prox(const Prox<T>& other) :
    index_(other.index_),
    size_(other.size_),
    diagsteps_(other.diagsteps_) { }
  
  virtual ~Prox();

  /**
   * @brief Initializes the prox Operator, copies data to the GPU.
   *
   */
  virtual void Init() { }

  /**
   * @brief Cleans up GPU data.
   *
   */
  virtual void Release() {}

  /**
   * @brief Evaluates the prox operator on the GPU. arg, result and
   *        tau are all pointers pointing to the whole vector in memory.
   *
   * @param Proximal operator argument.
   * @param Result of prox.
   * @param Scalar step size.
   * @param Diagonal step sizes.
   */
  void Eval(thrust::device_vector<T>& arg, thrust::device_vector<T>& res, thrust::device_vector<T>& tau_diag, T tau);
  
  void Eval(std::vector<T>& arg, std::vector<T>& res, std::vector<T>& tau_diag, T tau);
  // set/get methods
  virtual size_t gpu_mem_amount() = 0;  
  size_t index() const { return index_; }
  size_t size() const { return size_; }
  size_t end() const { return index_ + size_ - 1; }
  bool diagsteps() const { return diagsteps_; }
  
protected:
  /**
   * @brief Evaluates the prox operator on the GPU, local meaning that
   *        arg, res and tau point to the place in memory where the
   *        prox begins.
   *
   * @param Proximal operator argument.
   * @param Result of prox.
   * @param Scalar step size.
   * @param Diagonal step sizes.
   * @param Perform the prox with inverted step sizes?
   *
   */
  virtual void EvalLocal(const typename thrust::device_vector<T>::iterator& arg_begin,
                         const typename thrust::device_vector<T>::iterator& arg_end,
                         const typename thrust::device_vector<T>::iterator& res_begin,
                         const typename thrust::device_vector<T>::iterator& res_end,
                         const typename thrust::device_vector<T>::iterator& tau_begin,
                         const typename thrust::device_vector<T>::iterator& tau_end,
                         T tau,
                         bool invert_tau) = 0;
  
  size_t index_; 
  size_t size_;
  bool diagsteps_; // able to handle diagonal matrices as step size?
};
}
#endif
