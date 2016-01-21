#ifndef PROX_HPP_
#define PROX_HPP_

#include <stddef.h>
#include <thrust/device_vector.h>

template<typename T> class ProxMoreau;

/**
 * @brief Virtual base class for all proximal operators. 
 */
template<typename T>
class Prox {
  friend class ProxMoreau<T>;
  
public:
  Prox(size_t index, size_t size, bool diagsteps) :
    index_(index),
    size_(size),
    diagsteps_(diagsteps) {}

  Prox(const Prox<T>& other) :
    index_(other.index_),
    size_(other.size_),
    diagsteps_(other.diagsteps_) {}
  
  virtual ~Prox() {}

  /**
   * @brief Initializes the prox Operator, copies data to the GPU.
   *
   */
  virtual void Init() {}

  /**
   * @brief Cleans up GPU data.
   *
   */
  virtual void Release() {}

  /**
   * @brief Evaluates the prox operator on the GPU. 
   *
   * @param Proximal operator argument.
   * @param Result of prox.
   * @param Diagonal step sizes.
   * @param Scalar step size.
   */
  void Eval(
    thrust::device_vector<T>& result, 
    const thrust::device_vector<T>& arg, 
    const thrust::device_vector<T>& tau_diag, 
    T tau_scal);

  // set/get methods
  virtual size_t gpu_mem_amount() const = 0;
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
  virtual void EvalLocal(
    const thrust::device_ptr<T>& result,
    const thrust::device_ptr<const T>& arg,
    const thrust::device_ptr<const T>& tau_diag,
    T tau_scal,
    bool invert_tau) = 0;
  
  size_t index_; 
  size_t size_;
  bool diagsteps_; // able to handle diagonal matrices as step size?
};

#endif
