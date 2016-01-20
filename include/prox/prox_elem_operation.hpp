#ifndef PROX_ELEM_OPERATION_HPP_
#define PROX_ELEM_OPERATION_HPP_

#include <stddef.h>

#include "prox_separable_sum.hpp"
#include "shared_mem.hpp"


using namespace thrust;
using namespace std;

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
class ProxElemOperation : public ProxSeparableSum<T> {
public:    
  ProxElemOperation(size_t index, size_t count, interleaved, diagsteps, const vector<ELEM_OPERATION::Coefficients>& coeffs);
  
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

  // set/get methods
  virtual size_t gpu_mem_amount(); 
  
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
  virtual void EvalLocal(device_vector<T> d_arg,
                         device_vector<T> d_res,
                         device_vector<T> d_tau,
                         T tau,
                         bool invert_tau);
  
private:
  vector<ELEM_OPERATION::Coefficients> coeffs_;
  device_vector<ELEM_OPERATION::Coefficients> d_coeffs_;
};

#endif
