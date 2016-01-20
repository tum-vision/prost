#ifndef ELEM_OPERATION_SIMPLEX_HPP_
#define ELEM_OPERATION_SIMPLEX_HPP_

#include "elem_operation.hpp"


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
namespace elemOperation {
template<typename T, size_t DIM>
struct ElemOperationSimplex : public ElemOperation<DIM> {
  static const size_t shared_mem_count = DIM;
  typedef T shared_mem_type;
    
  struct Coefficients {
    T a[DIM];
  };
  
  ElemOperationSimplex(Coefficients& coeffs) : coeffs_(coeffs) {} 
  
  inline __device__ void operator()(Vector<T, ElemOperationSimplex>& arg, Vector<T, ElemOperationSimplex>& res, Vector<T, ElemOperationSimplex>& tau_diag, T tau_scal, bool invert_tau, SharedMem<ElemOperation1D>& shared_mem) {

      // 1) read dim-dimensional vector into shared memory
      for(size_t i = 0; i < dim; i++) {
        // handle inner product by completing the squaring and
        // pulling it into the squared term of the prox. while
        // scaling it correctly, taking are of the step size.
        T arg = d_arg[i];
        if(d_coeffs != NULL) {
          T tau_scaled = tau * d_tau[i];

          if(invert_tau)
            tau_scaled = 1. / tau_scaled;

          arg -= tau_scaled * coeffs_.a[i];
        }

        shared_mem[i] = arg;
      }
      __syncthreads();

      // 2) sort inside shared memory
      shellsort<T>(&shared_mem[0], dim);

      bool bget = false;
      T tmpsum = 0;
      T tmax;
      for(int ii=1;ii<=dim-1;ii++) {
        tmpsum += shared_mem[ii - 1];
        tmax = (tmpsum - 1.) / (T)ii;
        if(tmax >= shared_mem[ii]){
          bget=true;
          break;
        }
      }

      if(!bget)
        tmax = (tmpsum + shared_mem[dim - 1] - 1.0) / (T)dim;

      // 3) return result
      for(i = 0; i < dim; i++) {
        T arg = d_arg[i];
        if(d_coeffs != NULL) {
          T tau_scaled = tau * d_tau[i];

          if(invert_tau)
            tau_scaled = 1. / tau_scaled;

          arg -= tau_scaled * coeffs_[i];
        }

        d_res[i] = cuwrap::max<T>(arg - tmax, 0);
      }  
  }
  
  template<typename T>
  __device__ void shellsort(T *a, int N) {
      const int gaps[6] = { 132, 57, 23, 10, 4, 1 };

      for(int k = 0; k < 6; k++) {
        int gap = gaps[k];

        for(int i = gap; i < N; i++) {
          const T temp = a[i];

          for(int j = i; (j >= gap) && (a[j - gap] <= temp); j -= gap) 
            a[j] = a[j - gap];

          a[j] = temp;
        }
      }
  }  
};
}
}
#endif
