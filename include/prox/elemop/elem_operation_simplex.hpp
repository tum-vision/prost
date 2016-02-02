#ifndef ELEM_OPERATION_SIMPLEX_HPP_
#define ELEM_OPERATION_SIMPLEX_HPP_

#include "elem_operation.hpp"

namespace prox 
{

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
struct ElemOperationSimplex : public ElemOperation<0, 0, T> {

  #ifdef __CUDACC__
  inline __host__ __device__ static size_t GetSharedMemCount(size_t dim) { return dim; }

  __host__ __device__ ElemOperationSimplex(size_t dim, SharedMem<ElemOperationSimplex<T>>& shared_mem) : dim_(dim), shared_mem_(shared_mem) {} 
  
  inline __device__ void operator()(Vector<T>& arg, Vector<T>& res, Vector<T>& tau_diag, T tau_scal, bool invert_tau) 
  {
      // 1) read dim-dimensional vector into shared memory
      for(size_t i = 0; i < dim_; i++) {
        // handle inner product by completing the squaring and
        // pulling it into the squared term of the prox. while
        // scaling it correctly, taking are of the step size.
        T val = arg[i];

        shared_mem_[i] = val;
      }
      
      __syncthreads();
      

      // 2) sort inside shared memory
      shellsort();

      bool bget = false;
      T tmpsum = 0;
      T tmax;
      for(int ii=1; ii <= dim_ - 1; ii++) {
        tmpsum += shared_mem_[ii - 1];
        tmax = (tmpsum - 1.) / (T)ii;
        if(tmax >= shared_mem_[ii]){
          bget=true;
          break;
        }
      }

      if(!bget)
        tmax = (tmpsum + shared_mem_[dim_ - 1] - 1.0) / (T)dim_;

      // 3) return result
      for(int i = 0; i < dim_; i++) {
        T val = arg[i];

        res[i] = max(val - tmax, 0);
      }  
  }
  
  __device__ void shellsort() {
      const int gaps[6] = { 132, 57, 23, 10, 4, 1 };

      for(int k = 0; k < 6; k++) {
        int gap = gaps[k];

        for(int i = gap; i < dim_; i++) {
          const T temp = shared_mem_[i];

          int j = i;
          for(; (j >= gap) && (shared_mem_[j - gap] <= temp); j -= gap) 
            shared_mem_[j] = shared_mem_[j - gap];

          shared_mem_[j] = temp;
        }
      }
  }
  #endif
    
private:
  SharedMem<ElemOperationSimplex<T>>& shared_mem_;
  size_t dim_;
};

}

#endif