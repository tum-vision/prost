#ifndef ELEM_OPERATION_NORM2_HPP_
#define ELEM_OPERATION_NORM2_HPP_

#include "prox_1d.hpp"

/**
 * @brief Provides proximal operator for sum of 2-norms, with a nonlinear
 *        function ProxFunction1D applied to the norm.
 *
 *
 *
 */
namespace prox {
namespace elemOperation {
template<typename T, size_t DIM, OPERATION_1D>
struct ElemOperationNorm2 {
 struct Data {
    T a, b, c, d, e, alpha, beta;
 };
 
 static const size_t dim = DIM;

 virtual void operator()(ProxSeparableSum<T, ElemOperationNorm2>::Vector& arg, ProxSeparableSum<T, ElemOperationNorm2>::Vector& res, Data& data);
};
}
}
#endif
