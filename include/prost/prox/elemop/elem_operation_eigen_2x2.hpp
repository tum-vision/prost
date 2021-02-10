#ifndef PROST_ELEM_OPERATION_EIGEN_2X2_HPP_
#define PROST_ELEM_OPERATION_EIGEN_2X2_HPP_

#include "prost/prox/elemop/elem_operation.hpp"
#include <cfloat>


namespace prost {


    
    
    ///
    /// \brief Provides proximal operator for the indicator function of the cone of positive semidefinite matrices.
    /// The input has to be a vectorized real 3x3 matrix.
    template<typename T, class FUN_1D>
    struct ElemOperationEigen2x2 : public ElemOperation<4, 7>
    {
        
    __host__ __device__ ElemOperationEigen2x2(T* coeffs, size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) 
    : coeffs_(coeffs){ }
    
    // Macros
#define SQR(x)      ((x)*(x))                        // x^2 
#define SQR_ABS(x)  (SQR(creal(x)) + SQR(cimag(x)))  // |x|^2


// ----------------------------------------------------------------------------
inline __host__ __device__
        void dsyev2(double A, double B, double C, double *rt1, double *rt2,
                   double *cs, double *sn)
// ----------------------------------------------------------------------------
// Calculates the eigensystem of a real symmetric 2x2 matrix
//    [ A  B ]
//    [ B  C ]
// in the form
//    [ A  B ]  =  [ cs  -sn ] [ rt1   0  ] [  cs  sn ]
//    [ B  C ]     [ sn   cs ] [  0   rt2 ] [ -sn  cs ]
// where rt1 >= rt2. Note that this convention is different from the one used
// in the LAPACK routine DLAEV2, where |rt1| >= |rt2|.
// ----------------------------------------------------------------------------
{
  double sm = A + C;
  double df = A - C;
  double rt = sqrt(SQR(df) + 4.0*B*B);
  double t;

  if (sm > 0.0)
  {
    *rt1 = 0.5 * (sm + rt);
    t = 1.0/(*rt1);
    *rt2 = (A*t)*C - (B*t)*B;
  }
  else if (sm < 0.0)
  {
    *rt2 = 0.5 * (sm - rt);
    t = 1.0/(*rt2);
    *rt1 = (A*t)*C - (B*t)*B;
  }
  else       // This case needs to be treated separately to avoid div by 0
  {
    *rt1 = 0.5 * rt;
    *rt2 = -0.5 * rt;
  }

  // Calculate eigenvectors
  if (df > 0.0)
    *cs = df + rt;
  else
    *cs = df - rt;

  if (fabs(*cs) > 2.0*fabs(B))
  {
    t   = -2.0 * B / *cs;
    *sn = 1.0 / sqrt(1.0 + SQR(t));
    *cs = t * (*sn);
  }
  else if (fabs(B) == 0.0)
  {
    *cs = 1.0;
    *sn = 0.0;
  }
  else
  {
    t   = -0.5 * (*cs) / B;
    *cs = 1.0 / sqrt(1.0 + SQR(t));
    *sn = t * (*cs);
  }

  if (df > 0.0)
  {
    t   = *cs;
    *cs = -(*sn);
    *sn = t;
  }
}
        
        inline __host__ __device__
        void operator()(
                        Vector<T>& res,
                        const Vector<const T>& arg,
                        const Vector<const T>& tau_diag,
                        T tau_scal,
                        bool invert_tau)
        {
            

            double rt1, rt2;
            double cs, sn;
            
            dsyev2(arg[0], (arg[1]+arg[2]) / 2, arg[3], &rt1, &rt2,
                   &cs, &sn);
            
            // compute step-size
            double tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);
            
            if(coeffs_[0] == 0 || coeffs_[2] == 0) {
                rt1 = (rt1 - tau * coeffs_[3]) / (1 + tau * coeffs_[4]);
                rt2 = (rt2 - tau * coeffs_[3]) / (1 + tau * coeffs_[4]);
            } else {
                // compute scaled prox argument and step
                double p1, p2;
                p1 = ((coeffs_[0] * (rt1 - coeffs_[3] * tau)) / (1. + tau * coeffs_[4])) - coeffs_[1];
                p2 = ((coeffs_[0] * (rt2 - coeffs_[3] * tau)) / (1. + tau * coeffs_[4])) - coeffs_[1];
                
                const double step = (coeffs_[2] * coeffs_[0] * coeffs_[0] * tau) / (1. + tau * coeffs_[4]);

                // compute scaled prox and store result
                FUN_1D fun;
                rt1 = (fun(p1, step, coeffs_[5], coeffs_[6]) + coeffs_[1]) / coeffs_[0];
                rt2 = (fun(p2, step, coeffs_[5], coeffs_[6]) + coeffs_[1]) / coeffs_[0];
            }
               
            // compute T = V * S * V^T
            double t11 = rt1*SQR(cs) + rt2*SQR(sn);
            double t12 = rt1*cs*sn - sn*rt2*cs;
            double t22 = rt1*SQR(sn) + rt2*SQR(cs);

            res[0] = t11;
            res[1] = t12;
            res[2] = t12;
            res[3] = t22;
        }
        
        private:
            T* coeffs_;
    };
    
} // namespace prost

#endif // PROST_ELEM_OPERATION_EIGEN_2X2_HPP_

