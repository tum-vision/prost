#ifndef ELEM_OPERATION_SIMPLEX_HPP_
#define ELEM_OPERATION_SIMPLEX_HPP_

#include "elem_operation.hpp"
#include "../../util/cuwrap.hpp"


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
template<typename T, size_t DIM>
struct ElemOperationSimplex : public ElemOperation<DIM> {
  static const size_t shared_mem_count = DIM;
  typedef T shared_mem_type;
    
//  struct Coefficients {
//    T a[DIM];
//  };
  
//  __device__ ElemOperationSimplex(Coefficients& coeffs) : coeffs_(coeffs) {} 
   __device__ ElemOperationSimplex() {} 
  
  inline __device__ void operator()(Vector<T, ElemOperationSimplex<T, DIM>>& arg, Vector<T, ElemOperationSimplex<T, DIM>>& res, Vector<T, ElemOperationSimplex<T, DIM>>& tau_diag, T tau_scal, bool invert_tau, SharedMem<ElemOperationSimplex<T, DIM>>& shared_mem) {

      // 1) read dim-dimensional vector into shared memory
      for(size_t i = 0; i < DIM; i++) {
        // handle inner product by completing the squaring and
        // pulling it into the squared term of the prox. while
        // scaling it correctly, taking are of the step size.
        T val = arg[i];


//          T tau = tau_scal * tau_diag[i];

 //         if(invert_tau)
   //         tau = 1. / tau;

     //     val -= tau * coeffs_.a[i];


        shared_mem[i] = val;
      }
      __syncthreads();

      // 2) sort inside shared memory
      shellsort(&shared_mem[0], DIM);

      bool bget = false;
      T tmpsum = 0;
      T tmax;
      for(int ii=1;ii<=DIM-1;ii++) {
        tmpsum += shared_mem[ii - 1];
        tmax = (tmpsum - 1.) / (T)ii;
        if(tmax >= shared_mem[ii]){
          bget=true;
          break;
        }
      }

      if(!bget)
        tmax = (tmpsum + shared_mem[DIM - 1] - 1.0) / (T)DIM;

      // 3) return result
      for(int i = 0; i < DIM; i++) {
        T val = arg[i];

       //   T tau = tau_scal * tau_diag[i];

       //   if(invert_tau)
       //     tau = 1. / tau;

        //  val -= tau * coeffs_.a[i];


        res[i] = cuwrap::max<T>(val - tmax, 0);
      }  
  }
  
  __device__ void shellsort(T *a, int N) {
      const int gaps[6] = { 132, 57, 23, 10, 4, 1 };

      for(int k = 0; k < 6; k++) {
        int gap = gaps[k];

        for(int i = gap; i < N; i++) {
          const T temp = a[i];

          int j = i;
          for(; (j >= gap) && (a[j - gap] <= temp); j -= gap) 
            a[j] = a[j - gap];

          a[j] = temp;
        }
      }
  }
    
private:
  //Coefficients& coeffs_;
};
#endif
