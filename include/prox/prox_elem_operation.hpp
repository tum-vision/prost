#ifndef PROX_ELEM_OPERATION_HPP_
#define PROX_ELEM_OPERATION_HPP_

#include <stddef.h>

#include "prox_separable_sum.hpp"
#include "elemop/vector.hpp"
#include "elemop/elem_operation_1d.hpp"
#include "elemop/elem_operation_norm2.hpp"
#include "elemop/elem_operation_simplex.hpp"
#include "elemop/function_1d.hpp"
#include "../pdsolver_exception.hpp"

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
namespace prox {
template<typename T>
struct has_coeffs {
private:
    template<typename V> static void probe(decltype(typename V::Coefficients(), int()));
    template<typename V> static bool probe(char);
public:
    static const bool value = std::is_same<void, decltype(probe<T>(0))>::value;
};  

template<typename T, class ELEM_OPERATION, class ENABLE = void>
class ProxElemOperation {};

template<typename T, class ELEM_OPERATION>
class ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<!has_coeffs<ELEM_OPERATION>::value>::type> : public ProxSeparableSum<T> {
public:    
  ProxElemOperation(size_t index, size_t count, size_t dim, bool interleaved, bool diagsteps) : ProxSeparableSum<T>(index, count, ELEM_OPERATION::dim <= 0 ? dim : ELEM_OPERATION::dim, interleaved, diagsteps) {}

  /**
   * @brief Initializes the prox Operator, copies data to the GPU.
   *
   */
  virtual void Init() {
  }

  virtual void Release();
  
  // set/get methods
  virtual size_t gpu_mem_amount() {
    return 0;
  }
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
                         bool invert_tau);
};

template<typename T, class ELEM_OPERATION>
class ProxElemOperation<T, ELEM_OPERATION, typename std::enable_if<has_coeffs<ELEM_OPERATION>::value>::type> : public ProxSeparableSum<T> {
public:    
  ProxElemOperation(size_t index, size_t count, size_t dim, bool interleaved, bool diagsteps, const std::vector<typename ELEM_OPERATION::Coefficients>& coeffs) : ProxSeparableSum<T>(index, count, ELEM_OPERATION::dim == 0 ? dim : ELEM_OPERATION::dim, interleaved, diagsteps), coeffs_(coeffs) {}

  /**
   * @brief Initializes the prox Operator, copies data to the GPU.
   *
   */
  virtual void Init() {
    try {
        thrust::copy(coeffs_.begin(), coeffs_.end(), d_coeffs_.begin());
    } catch(std::bad_alloc &e) {
        throw PDSolverException();
    } catch(thrust::system_error &e) {
        throw PDSolverException();
    }
  }

  virtual void Release();
  
  // set/get methods
  virtual size_t gpu_mem_amount() {
    return this->count_ * sizeof(typename ELEM_OPERATION::Coefficients);
  }

   
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
  virtual void EvalLocal(const typename thrust::device_vector<T>::iterator& arg_begin,
                         const typename thrust::device_vector<T>::iterator& arg_end,
                         const typename thrust::device_vector<T>::iterator& res_begin,
                         const typename thrust::device_vector<T>::iterator& res_end,
                         const typename thrust::device_vector<T>::iterator& tau_begin,
                         const typename thrust::device_vector<T>::iterator& tau_end,
                         T tau,
                         bool invert_tau);
  
private:
  std::vector<typename ELEM_OPERATION::Coefficients> coeffs_;
  thrust::device_vector<typename ELEM_OPERATION::Coefficients> d_coeffs_;  
};
}
#endif
