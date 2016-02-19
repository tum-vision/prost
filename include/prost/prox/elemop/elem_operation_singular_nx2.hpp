#ifndef PROST_ELEM_OPERATION_SINGULAR_NX2_HPP_
#define PROST_ELEM_OPERATION_SINGULAR_NX2_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {

/// 
/// \brief Provides proximal operator for singular values of a Nx2 matrix, 
///        with a lower semicontinuous function FunctionNd applied to the singular values.
/// 
template<typename T, class FUN_2D>
struct ElemOperationSingularNx2 : public ElemOperation<6, 7> 
{
  __host__ __device__ 
  ElemOperationSingularNx2(T* coeffs, size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) 
    : coeffs_(coeffs), dim_(dim) { } 
 
 inline __host__ __device__ 
 void operator()(
     Vector<T>& res, 
     const Vector<const T>& arg, 
     const Vector<const T>& tau_diag, 
     T tau_scal, 
     bool invert_tau) 
  {
    size_t n = dim_ / 2;
    // compute D = A^T A
    T d11 = 0., d12 = 0., d22 = 0.;
    
    for(size_t i = 0; i < n; i++) {
      d11 += arg[i] * arg[i];
      d12 += arg[i] * arg[n+i];
      d22 += arg[n+i] * arg[n+i];
    }

    // compute eigenvalues of (lmax, lmin)
    T trace = d11 + d22;
    T det = d11*d22 - d12*d12;
    T d = sqrt(max(static_cast<T>(0), static_cast<T>(0.25)*trace*trace - det));
    T lmax = max(static_cast<T>(0), static_cast<T>(0.5) * trace + d);
    T lmin = max(static_cast<T>(0), static_cast<T>(0.5) * trace - d);
    T smax = sqrt(lmax);
    T smin = sqrt(lmin);

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

    if(smax > 0) {
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

      // Compute \Sigma^+ * \Sigma p 
      s1 /= smax;
      s2 = (smin > 0.0) ? (s2 / smin) : 0.0;

      // Compute T = V * \Sigma^+ * \Sigma_p * VˆT

      T t11 = s1*v11*v11 + s2*v12*v12;
      T t12 = s1*v11*v21 + s2*v12*v22;
      T t21 = s1*v21*v11 + s2*v22*v12;
      T t22 = s1*v21*v21 + s2*v22*v22;

      for(size_t i = 0; i < n; i++) {
        res[i] = arg[i] * t11 + arg[n+i] * t21;
        res[n+i] = arg[i] * t12 + arg[n+i] * t22;
      }
    }
    else {
      for(size_t i = 0; i < 2*n; i++) {
          res[i] = 0;
      }
      res[0] = s1;
      res[n+1] = s2;
    }
  }

 
private:
  size_t dim_;
  T* coeffs_;
};

} // namespace prost

#endif // PROST_ELEM_OPERATION_SINGULAR_NX2_HPP_
