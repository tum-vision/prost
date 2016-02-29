#ifndef PROST_ELEM_OPERATION_IND_SIMPLEX_HPP_
#define PROST_ELEM_OPERATION_IND_SIMPLEX_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {

#define MAX_DIM 1024

/// 
/// \brief Computes prox for sum of simplex indicator functions.
///
///        sum_{i=1}^{count} delta_dim(x_i) + <x_i, a_i>,
/// 
///        where delta_dim denotes the dim-dimensional simplex.
///        See http://arxiv.org/pdf/1101.6081v2.pdf.
///
///        Replaced shared memory by local memory. (cached on newer architectures)
///
template<typename T>
struct ElemOperationIndSimplex : public ElemOperation<0, 0, T>
{
  __device__
  ElemOperationIndSimplex(size_t dim, SharedMem<typename ElemOperationIndSimplex::SharedMemType, typename ElemOperationIndSimplex::GetSharedMemCount>& shared_mem)
      : dim_(dim) { } 
  
  inline __device__
  void
  operator()(
    Vector<T>& res,
    const Vector<const T>& arg,
    const Vector<const T>& tau_diag,
    T tau_scal,
    bool invert_tau) 
  {
    T local_mem[MAX_DIM]; 

    // 1) read dim-dimensional vector into local memory
    for(size_t i = 0; i < dim_; i++)
    {
      T val = arg[i];
      local_mem[i] = val;
    }
           
    // 2) sort inside local memory
    ShellSort(local_mem);

    bool bget = false;
    T tmpsum = 0;
    T tmax;
    for(int ii=1; ii <= dim_ - 1; ii++)
    {
      tmpsum += local_mem[ii - 1];
      tmax = (tmpsum - 1.) / (T)ii;
      if(tmax >= local_mem[ii])
      {
        bget=true;
        break;
      }
    }

    if(!bget)
      tmax = (tmpsum + local_mem[dim_ - 1] - 1.0) / (T)dim_;

    // 3) return result
    for(int i = 0; i < dim_; i++)
    {
      T val = arg[i];

      res[i] = max(val - tmax, static_cast<T>(0));
    }  
  }
  
  __device__
  void
  ShellSort(T *array)
  {
    const int gaps[6] = { 132, 57, 23, 10, 4, 1 };

    for(int k = 0; k < 6; k++)
    {
      int gap = gaps[k];

      for(int i = gap; i < dim_; i++)
      {
        const T temp = array[i];

        int j = i;
        for(; (j >= gap) && (array[j - gap] <= temp); j -= gap) 
          array[j] = array[j - gap];

        array[j] = temp;
      }
    }
  }
    
 private:
  size_t dim_;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_SIMPLEX_HPP_
