#ifndef PROST_ELEM_OPERATION_EIGEN_3X3_HPP_
#define PROST_ELEM_OPERATION_EIGEN_3X3_HPP_

#include "prost/prox/elemop/elem_operation.hpp"
#include <cfloat>


namespace prost {


    
    
    ///
    /// \brief Provides proximal operator for the indicator function of the cone of positive semidefinite matrices.
    /// The input has to be a vectorized real 3x3 matrix.
    template<typename T, class FUN_1D>
    struct ElemOperationEigen3x3 : public ElemOperation<9, 7>
    {
        
    __host__ __device__ ElemOperationEigen3x3(T* coeffs, size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) 
    : coeffs_(coeffs){ } 
        
        
        
// Constants
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

// Macros
#define SQR(x)      ((x)*(x))                        // x^2 


// ----------------------------------------------------------------------------
__host__ __device__
int dsyevc3(double A[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
// analytical algorithm.
// Only the diagonal and upper triangular parts of A are accessed. The access
// is read-only.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
{
  double m, c1, c0;
  
  // Determine coefficients of characteristic poynomial. We write
  //       | a   d   f  |
  //  A =  | d*  b   e  |
  //       | f*  e*  c  |
  double de = A[0][1] * A[1][2];                                    // d * e
  double dd = SQR(A[0][1]);                                         // d^2
  double ee = SQR(A[1][2]);                                         // e^2
  double ff = SQR(A[0][2]);                                         // f^2
  m  = A[0][0] + A[1][1] + A[2][2];
  c1 = (A[0][0]*A[1][1] + A[0][0]*A[2][2] + A[1][1]*A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
          - (dd + ee + ff);
  c0 = A[2][2]*dd + A[0][0]*ee + A[1][1]*ff - A[0][0]*A[1][1]*A[2][2]
            - 2.0 * A[0][2]*de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

  double p, sqrt_p, q, c, s, phi;
  p = SQR(m) - 3.0*c1;
  q = m*(p - (3.0/2.0)*c1) - (27.0/2.0)*c0;
  sqrt_p = sqrt(fabs(p));

  phi = 27.0 * ( 0.25*SQR(c1)*(p - c1) + c0*(q + 27.0/4.0*c0));
  phi = (1.0/3.0) * atan2(sqrt(fabs(phi)), q);
  
  c = sqrt_p*cos(phi);
  s = (1.0/M_SQRT3)*sqrt_p*sin(phi);

  w[1]  = (1.0/3.0)*(m - c);
  w[2]  = w[1] + s;
  w[0]  = w[1] + c;
  w[1] -= s;

  return 0;
}

// ----------------------------------------------------------------------------
__host__ __device__
int dsyevv3(double A[3][3], double Q[3][3], double w[3])
// ----------------------------------------------------------------------------
// Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
// matrix A using Cardano's method for the eigenvalues and an analytical
// method based on vector cross products for the eigenvectors.
// Only the diagonal and upper triangular parts of A need to contain meaningful
// values. However, all of A may be used as temporary storage and may hence be
// destroyed.
// ----------------------------------------------------------------------------
// Parameters:
//   A: The symmetric input matrix
//   Q: Storage buffer for eigenvectors
//   w: Storage buffer for eigenvalues
// ----------------------------------------------------------------------------
// Return value:
//   0: Success
//  -1: Error
// ----------------------------------------------------------------------------
// Dependencies:
//   dsyevc3()
// ----------------------------------------------------------------------------
// Version history:
//   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//     (according to the documentation, only the upper triangular part needs
//     to be filled)
//   v1.0: First released version
// ----------------------------------------------------------------------------
{
  double norm;          // Squared norm or inverse norm of current eigenvector
  double n0, n1;        // Norm of first and second columns of A
  double n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
  double thresh;        // Small number used as threshold for floating point comparisons
  double error;         // Estimated maximum roundoff error in some steps
  double wmax;          // The eigenvalue of maximum modulus
  double f, t;          // Intermediate storage
  int i, j;             // Loop counters

  // Calculate eigenvalues
  dsyevc3(A, w);

  wmax = fabs(w[0]);
  if ((t=fabs(w[1])) > wmax)
    wmax = t;
  if ((t=fabs(w[2])) > wmax)
    wmax = t;
  thresh = SQR(8.0 * DBL_EPSILON * wmax);

  // Prepare calculation of eigenvectors
  n0tmp   = SQR(A[0][1]) + SQR(A[0][2]);
  n1tmp   = SQR(A[0][1]) + SQR(A[1][2]);
  Q[0][1] = A[0][1]*A[1][2] - A[0][2]*A[1][1];
  Q[1][1] = A[0][2]*A[0][1] - A[1][2]*A[0][0];
  Q[2][1] = SQR(A[0][1]);

  // Calculate first eigenvector by the formula
  //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
  A[0][0] -= w[0];
  A[1][1] -= w[0];
  Q[0][0] = Q[0][1] + A[0][2]*w[0];
  Q[1][0] = Q[1][1] + A[1][2]*w[0];
  Q[2][0] = A[0][0]*A[1][1] - Q[2][1];
  norm    = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
  n0      = n0tmp + SQR(A[0][0]);
  n1      = n1tmp + SQR(A[1][1]);
  error   = n0 * n1;
  
  if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
  {
    Q[0][0] = 1.0;
    Q[1][0] = 0.0;
    Q[2][0] = 0.0;
  }
  else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
  {
    Q[0][0] = 0.0;
    Q[1][0] = 1.0;
    Q[2][0] = 0.0;
  }
  else if (norm < SQR(64.0 * DBL_EPSILON) * error)
  {                         // If angle between A[0] and A[1] is too small, don't use
    t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
    f = -A[0][0] / A[0][1];
    if (SQR(A[1][1]) > t)
    {
      t = SQR(A[1][1]);
      f = -A[0][1] / A[1][1];
    }
    if (SQR(A[1][2]) > t)
      f = -A[0][2] / A[1][2];
    norm    = 1.0/sqrt(1 + SQR(f));
    Q[0][0] = norm;
    Q[1][0] = f * norm;
    Q[2][0] = 0.0;
  }
  else                      // This is the standard branch
  {
    norm = sqrt(1.0 / norm);
    for (j=0; j < 3; j++)
      Q[j][0] = Q[j][0] * norm;
  }

  
  // Prepare calculation of second eigenvector
  t = w[0] - w[1];
  if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
  {
    // For non-degenerate eigenvalue, calculate second eigenvector by the formula
    //   v[1] = (A - w[1]).e1 x (A - w[1]).e2
    A[0][0] += t;
    A[1][1] += t;
    Q[0][1]  = Q[0][1] + A[0][2]*w[1];
    Q[1][1]  = Q[1][1] + A[1][2]*w[1];
    Q[2][1]  = A[0][0]*A[1][1] - Q[2][1];
    norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
    n0       = n0tmp + SQR(A[0][0]);
    n1       = n1tmp + SQR(A[1][1]);
    error    = n0 * n1;
 
    if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
    {
      Q[0][1] = 1.0;
      Q[1][1] = 0.0;
      Q[2][1] = 0.0;
    }
    else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
    {
      Q[0][1] = 0.0;
      Q[1][1] = 1.0;
      Q[2][1] = 0.0;
    }
    else if (norm < SQR(64.0 * DBL_EPSILON) * error)
    {                       // If angle between A[0] and A[1] is too small, don't use
      t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
      f = -A[0][0] / A[0][1];
      if (SQR(A[1][1]) > t)
      {
        t = SQR(A[1][1]);
        f = -A[0][1] / A[1][1];
      }
      if (SQR(A[1][2]) > t)
        f = -A[0][2] / A[1][2];
      norm    = 1.0/sqrt(1 + SQR(f));
      Q[0][1] = norm;
      Q[1][1] = f * norm;
      Q[2][1] = 0.0;
    }
    else
    {
      norm = sqrt(1.0 / norm);
      for (j=0; j < 3; j++)
        Q[j][1] = Q[j][1] * norm;
    }
  }
  else
  {
    // For degenerate eigenvalue, calculate second eigenvector according to
    //   v[1] = v[0] x (A - w[1]).e[i]
    //   
    // This would really get to complicated if we could not assume all of A to
    // contain meaningful values.
    A[1][0]  = A[0][1];
    A[2][0]  = A[0][2];
    A[2][1]  = A[1][2];
    A[0][0] += w[0];
    A[1][1] += w[0];
    for (i=0; i < 3; i++)
    {
      A[i][i] -= w[1];
      n0       = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
      if (n0 > thresh)
      {
        Q[0][1]  = Q[1][0]*A[2][i] - Q[2][0]*A[1][i];
        Q[1][1]  = Q[2][0]*A[0][i] - Q[0][0]*A[2][i];
        Q[2][1]  = Q[0][0]*A[1][i] - Q[1][0]*A[0][i];
        norm     = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
        if (norm > SQR(256.0 * DBL_EPSILON) * n0) // Accept cross product only if the angle between
        {                                         // the two vectors was not too small
          norm = sqrt(1.0 / norm);
          for (j=0; j < 3; j++)
            Q[j][1] = Q[j][1] * norm;
          break;
        }
      }
    }
    
    if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
    {
      for (j=0; j < 3; j++)
        if (Q[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
        {                                                     // ... and swap it with the next one
          norm          = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j+1)%3][0]));
          Q[j][1]       = Q[(j+1)%3][0] * norm;
          Q[(j+1)%3][1] = -Q[j][0] * norm;
          Q[(j+2)%3][1] = 0.0;
          break;
        }
    }
  }
      
  
  // Calculate third eigenvector according to
  //   v[2] = v[0] x v[1]
  Q[0][2] = Q[1][0]*Q[2][1] - Q[2][0]*Q[1][1];
  Q[1][2] = Q[2][0]*Q[0][1] - Q[0][0]*Q[2][1];
  Q[2][2] = Q[0][0]*Q[1][1] - Q[1][0]*Q[0][1];

  return 0;
}
        

        inline __host__ __device__
        void operator()(
                        Vector<T>& res,
                        const Vector<const T>& arg,
                        const Vector<const T>& tau_diag,
                        T tau_scal,
                        bool invert_tau)
        {
            
            double A[3][3];
            A[0][0] = arg[0];
            
            A[1][0] = (arg[1] + arg[3]) / 2.;
            A[0][1] = A[1][0];
            
            A[2][0] = (arg[2] + arg[6]) / 2;
            A[0][2] = A[2][0];
            
            A[1][1] = arg[4];
            
            A[2][1] = (arg[5] + arg[7]) / 2;
            A[1][2] = A[2][1];
            
            A[2][2] = arg[8];
            
            double V[3][3];
            double eig[3];
            
            dsyevv3(A, V, eig);
            
            // compute step-size
            double tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);
            
            if(coeffs_[0] == 0 || coeffs_[2] == 0) {
                eig[0] = (eig[0] - tau * coeffs_[3]) / (1 + tau * coeffs_[4]);
                eig[1] = (eig[1] - tau * coeffs_[3]) / (1 + tau * coeffs_[4]);
                eig[2] = (eig[2] - tau * coeffs_[3]) / (1 + tau * coeffs_[4]);
            } else {
                // compute scaled prox argument and step
                double prox_arg[3];
                prox_arg[0] = ((coeffs_[0] * (eig[0] - coeffs_[3] * tau)) / (1. + tau * coeffs_[4])) - coeffs_[1];
                prox_arg[1] = ((coeffs_[0] * (eig[1] - coeffs_[3] * tau)) / (1. + tau * coeffs_[4])) - coeffs_[1];
                prox_arg[2] = ((coeffs_[0] * (eig[2] - coeffs_[3] * tau)) / (1. + tau * coeffs_[4])) - coeffs_[1];
                
                const double step = (coeffs_[2] * coeffs_[0] * coeffs_[0] * tau) / (1. + tau * coeffs_[4]);

                // compute scaled prox and store result
                FUN_1D fun;
                eig[0] = (fun(prox_arg[0], step, coeffs_[5], coeffs_[6]) + coeffs_[1]) / coeffs_[0];
                eig[1] = (fun(prox_arg[1], step, coeffs_[5], coeffs_[6]) + coeffs_[1]) / coeffs_[0];
                eig[2] = (fun(prox_arg[2], step, coeffs_[5], coeffs_[6]) + coeffs_[1]) / coeffs_[0];
            }
               
            // compute T = V * S * V^T
            double t11 = V[0][0] * V[0][0] * eig[0] + V[0][1] * V[0][1] * eig[1] + V[0][2] * V[0][2] * eig[2];
            double t12 = V[0][0] * V[1][0] * eig[0] + V[0][1] * V[1][1] * eig[1] + V[0][2] * V[1][2] * eig[2];
            double t13 = V[0][0] * V[2][0] * eig[0] + V[0][1] * V[2][1] * eig[1] + V[0][2] * V[2][2] * eig[2];
            
//            double t21 = V[1][0] * V[0][0] * eig[0] + V[1][1] * V[0][1] * eig[1] + V[1][2] * V[0][2] * eig[2];
            double t22 = V[1][0] * V[1][0] * eig[0] + V[1][1] * V[1][1] * eig[1] + V[1][2] * V[1][2] * eig[2];
            double t23 = V[1][0] * V[2][0] * eig[0] + V[1][1] * V[2][1] * eig[1] + V[1][2] * V[2][2] * eig[2];
            
//            double t31 = V[2][0] * V[0][0] * eig[0] + V[2][1] * V[0][1] * eig[1] + V[2][2] * V[0][2] * eig[2];
//            double t32 = V[2][0] * V[1][0] * eig[0] + V[2][1] * V[1][1] * eig[1] + V[2][2] * V[1][2] * eig[2];
            double t33 = V[2][0] * V[2][0] * eig[0] + V[2][1] * V[2][1] * eig[1] + V[2][2] * V[2][2] * eig[2];
            
            res[0] = t11;
            res[1] = t12;
            res[2] = t13;
            
            res[3] = t12;
            res[4] = t22;
            res[5] = t23;
            
            res[6] = t13;
            res[7] = t23;
            res[8] = t33;
        }
        
        private:
            T* coeffs_;
    };
    
} // namespace prost

#endif // PROST_ELEM_OPERATION_EIGEN_3X3_HPP_

