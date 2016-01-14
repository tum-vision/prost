#ifndef PROX_NORM2_HPP_
#define PROX_NORM2_HPP_

#include "prox_1d.hpp"

/**
 * @brief Provides proximal operator for sum of 2-norms, with a nonlinear
 *        function ProxFunction1D applied to the norm.
 *
 *
 *
 */
namespace prox {
template<typename T, size_t DIM>
struct ElemOperationNorm2 {
 struct Data {
    T a[DIM];
    T c, d;
 };

 virtual void operator()(T* arg, T* res, Data &data);
};
}
#endif
