#ifndef ELEM_OPERATION_SIMPLEX_HPP_
#define ELEM_OPERATION_SIMPLEX_HPP_

#include <vector>
#include "prox.hpp"

/**
 * @brief Computes prox for sum of simplex indicator functions
 *        plus a linear term:
 *
 *        sum_{i=1}^{count} delta_dim(x_i) + <x_i, a_i>,
 *
 *        where delta_dim denotes the dim-dimensional simplex.
 *        See http://arxiv.org/pdf/1101.6081v2.pdf.
 *
 *        WARNING: Only efficient for small values of dim, because
 *        of limited amount of shared memory on GPUs. Might not work
 *        for too high values of dim (>32) or (>16 with double precision)
 *        because there's not enough shared mem. Sorting in global mem
 *        would be much too slow.
 */
namespace prox {
template<typename T, size_t DIM>
struct ElemOperationSimplex {
 struct Data {
    T a[DIM];
    T c, d;
 };

 virtual void operator()(T* arg, T* res, Data &data);
};
}

#endif
