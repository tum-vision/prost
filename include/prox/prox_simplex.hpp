#ifndef PROX_SIMPLEX_HPP_
#define PROX_SIMPLEX_HPP_

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
template<typename T>
class ProxSimplex : public Prox<T> {
 public:
  ProxSimplex(size_t index,
              size_t count,
              size_t dim,
              bool interleaved,
              const std::vector<T>& coeffs);

  virtual ~ProxSimplex();

  virtual bool Init();
  virtual void Release();
  virtual size_t gpu_mem_amount();

 protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);

protected:
  std::vector<T> coeffs_;
  T *d_coeffs_;
};

#endif
