#ifndef PROST_ELEM_OPERATION_SINGULAR_3X2_HPP_
#define PROST_ELEM_OPERATION_SINGULAR_3X2_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {

/// 
/// \brief Provides proximal operator for singular values of a 3x2 matrix, 
///        with a lower semicontinuous function FunctionNd applied to the singular values.
/// 
template<typename T, class FUN_2D>
struct ElemOperationSingular3x2 : public ElemOperation<6, 7> 
{
  __host__ __device__ 
  ElemOperationSingular3x2(T* coeffs, size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) 
    : coeffs_(coeffs) { } 
 
 inline __host__ __device__ 
 void operator()(
     Vector<T>& res, 
     const Vector<const T>& arg, 
     const Vector<const T>& tau_diag, 
     T tau_scal, 
     bool invert_tau) 
  {
    // compute D = A^T A
    T d11 = arg[0] * arg[0] + arg[1] * arg[1] + arg[2] * arg[2];
    T d12 = arg[3] * arg[0] + arg[4] * arg[1] + arg[5] * arg[2];
    T d22 = arg[3] * arg[3] + arg[4] * arg[4] + arg[5] * arg[5];

    // compute eigenvalues of (lmax, lmin)
    T trace = d11 + d22;
    T det = d11*d22 - d12*d12;
    T d = sqrt(max(0.0, 0.25*trace*trace - det));
    T lmax = max(0.0, 0.5 * trace + d);
    T lmin = max(0.0, 0.5 * trace - d);
    T smax = sqrt(lmax);
    T smin = sqrt(lmin);


    // Compute orthonormal system of Eigenvectors , such that
    // (v11, v21) belongs to lmax and (v12, v22) belongs to lmin
    T v11, v12, v21, v22;
    if(d12 == 0.0) {
      if (d11 >= d22) {
	    v11 = 1.0;
        v21 = 0.0;
        v12 = 0.0;
        v22 = 1.0;
      } else {
        v11 = 0.0;
        v21 = 1.0;
        v12 = 1.0;
        v22 = 0.0;
      }
    } else {
      v11 = lmax - d22;
      v21 = d12;
      T l1 = hypot(v11, v21);
      v11 /= l1;
      v21 /= l1;
      v12 = lmin - d22;
      v22 = d12;
      T l2 = hypot(v12, v22);
      v12 /= l2;
      v22 /= l2; 
    } 

    // Project (smax, smin) 

    // compute step-size
    T tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);

    T s1, s2;
    if(coeffs_[0] == 0 || coeffs_[2] == 0) {
      s1 = (smax - tau * coeffs_[3]) / (1. + tau * coeffs_[4]);
      s2 = (smin - tau * coeffs_[3]) / (1. + tau * coeffs_[4]);
    } else {
      // compute scaled prox argument and step
      T y1 = ((coeffs_[0] * (smax - coeffs_[3] * tau)) /
            (1. + tau * coeffs_[4])) - coeffs_[1];
      T y2 = ((coeffs_[0] * (smin - coeffs_[3] * tau)) /
          (1. + tau * coeffs_[4])) - coeffs_[1];

      const T step = (coeffs_[2] * coeffs_[0] * coeffs_[0] * tau) /
            (1. + tau * coeffs_[4]);


      // compute scaled prox and store result
      FUN_2D fun;
      T x1, x2;
      fun(y1, y2, x1, x2, step, coeffs_[5], coeffs_[6]);

      s1 =
          (x1 + coeffs_[1])
          / coeffs_[0];
      s2 =
          (x2 + coeffs_[1])
          / coeffs_[0];
    }

    // Compute \Sigma^+ * \Sigma p 
    s1 /= smax;
    s2 = (smin > 0.0) ? (s2 / smin) : 0.0;

    // Compute T = V * \Sigma^+ * \Sigma_p * VˆT

    T t11 = s1*v11*v11 + s2*v12*v12;
    T t12 = s1*v11*v21 + s2*v12*v22;
    T t21 = s1*v21*v11 + s2*v22*v12;
    T t22 = s1*v21*v21 + s2*v22*v22;

    res[0] = arg[0] * t11 + arg[3] * t21;
    res[1] = arg[1] * t11 + arg[4] * t21;
    res[2] = arg[2] * t11 + arg[5] * t21;
    res[3] = arg[0] * t12 + arg[3] * t22;
    res[4] = arg[1] * t12 + arg[4] * t22;
    res[5] = arg[2] * t12 + arg[5] * t22;
  }

 
private:
  T* coeffs_;
  size_t dim_;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_SINGULAR_3X2_HPP_
