#ifndef PROST_ELEM_OPERATION_IND_EPI_CONJQUAD_HPP_
#define PROST_ELEM_OPERATION_IND_EPI_CONJQUAD_HPP__

#include "prost/prox/elemop/elem_operation.hpp"
#include "prost/math_functions.hpp"

namespace prost {

/// 
/// \brief Provides proximal operator for sum of indicator functions 
///        of the epigraph of the function (ax^2 + bx + c + delta(alpha <= x <= beta))*,
///        where * denotes the Legendre-Fenchel conjugate.
/// 
///        For a detailed explanation see left part of Fig. 4 in the paper
///        http://arxiv.org/abs/1512.01383
/// 
template<typename T>
struct ElemOperationIndEpiConjQuad : public ElemOperation<2, 5> 
{
  __host__ __device__ 
  ElemOperationIndEpiConjQuad(T* coeffs, size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) 
    : coeffs_(coeffs) { } 
 
  inline __host__ __device__ 
  void operator()(
    Vector<T>& res, 
    const Vector<T>& arg, 
    const Vector<T>& tau_diag, 
    T tau_scal, 
    bool invert_tau) 
  {
    const T a = coeffs_[0];
    const T b = coeffs_[1];
    const T c = coeffs_[2];
    const T alpha = coeffs_[3];
    const T beta = coeffs_[4];
    T v[2] = { arg[0], arg[1] };
    T result[2];

    // compute function value to see if we are already in epigraph
    T fun_val;
    if(a > 0) 
    {
      if(v[0] < 2. * a * alpha + b)
        fun_val = alpha * v[0] - a * alpha * alpha - b * alpha - c;
      else if(v[0] > 2. * a * beta + b)
        fun_val = beta * v[0] - a * beta * beta - b * beta - c;
      else
        fun_val = v[0] * v[0] / (4. * a) - b * v[0] / (2. * a) + b * b / (4. * a) - c;
    }
    else {
      if(v[0] < a * (alpha + beta) + b) {
        fun_val = alpha * v[0] - a * alpha * alpha - b * alpha - c;
      }
      else {
        fun_val = beta * v[0] - a * beta * beta - b * beta - c;
      }
    }
    
    // check if we are in/on the epigraph
    if(arg[1] >= fun_val) 
    {
      // nothing to do!
      res[0] = v[0];
      res[1] = v[1];
    }
    else {
      // check which case applies (0 = A, 1 = B, 2 = C)

      // compute boundary points between A, B and C
      T p_A[2]; // point on epigraph of boundary between A and B
      T p_B[2]; // point on epigraph of boundary between B and C
      if(a < 0) 
      {
        p_A[0] = p_B[0] = a * (alpha + beta) + b;
        p_A[1] = p_B[1] = alpha * beta * a - c;
      }
      else 
      {
        p_A[0] = 2 * a * alpha + b;
        p_A[1] = a * alpha * alpha - c;
        p_B[0] = 2 * a * beta + b;
        p_B[1] = a * beta * beta - c;
      }

      // normals of the halfspaces A and B
      T n_A[2] = { 1, alpha };
      T n_B[2] = { -1, -beta }; 
    
      int proj_case;
      if(PointInHalfspace(v, p_A, n_A, 2))
        proj_case = 0;
      else if(PointInHalfspace(v, p_B, n_B, 2))
        proj_case = 2;
      else
        proj_case = 1;
    
      // perform projection
      switch(proj_case) 
      {
        case 0: { // case A
          n_A[0] = -alpha;
          n_A[1] = 1.;
          const T t = -a * alpha * alpha - b * alpha - c;
          
          ProjectHalfspace<T>(v,
                              n_A,
                              t,
                              result,
                              2);
        } break;

        case 1: { // case B
          if(a > 0) 
            ProjectParabolaGeneral<T>(v[0],
                                      v[1],
                                      1. / (4. * a),
                                      -b / (2. * a),
                                      b * b / (4. * a) - c,
                                      result[0],
                                      result[1]);
          else {
            // if a <= 0 the parabola disappears and we're in the normal cone.
            result[0] = a * (alpha + beta) + b;
            result[1] = alpha * beta * a - c;
          }
          
        } break;

        case 2: { // case C
          n_B[0] = -beta;
          n_B[1] = 1.;
          const T t = -a * beta * beta - b * beta - c;

          ProjectHalfspace<T>(v,
                              n_B,
                              t,
                              result,
                              2);
        } break;
      }
    }
    
    res[0] = result[0];
    res[1] = result[1];
  }
 
private:
  T* coeffs_;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_NORM2_HPP_
