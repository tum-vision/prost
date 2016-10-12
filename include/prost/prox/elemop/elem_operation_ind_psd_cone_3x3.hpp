#ifndef PROST_ELEM_OPERATION_IND_PSD_CONE_3X3_HPP_
#define PROST_ELEM_OPERATION_IND_PSD_CONE_3X3_HPP_

#include "prost/prox/elemop/elem_operation.hpp"

namespace prost {


    
    
    ///
    /// \brief Provides proximal operator for the indicator function of the cone of positive semidefinite matrices.
    /// The input has to be a real symmetric 3x3 matrix, where the lower triangluar matrix is handed over as a 6d vector.
    ///
    template<typename T>
    struct ElemOperationIndPsdCone3x3 : public ElemOperation<6, 0>
    {
        
        __host__ __device__
        ElemOperationIndPsdCone3x3(size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) { }
        
#ifdef MAX
#undef MAX
#endif
        
#define MAX(a, b) ((a)>(b)?(a):(b))
        
        __host__ __device__
        double myhypot2(double x, double y) {
            return sqrt(x*x+y*y);
        }
        
        // Symmetric Householder reduction to tridiagonal form.
        
        __host__ __device__
        void mytred2(double V[3][3], double d[3], double e[3]) {
            
            const int n=3;
            //  This is derived from the Algol procedures mytred2 by
            //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            //  Fortran subroutine in EISPACK.
            
            for (int j = 0; j < n; j++) {
                d[j] = V[n-1][j];
            }
            
            // Householder reduction to tridiagonal form.
            
            for (int i = n-1; i > 0; i--) {
                
                // Scale to avoid under/overflow.
                
                double scale = 0.0;
                double h = 0.0;
                for (int k = 0; k < i; k++) {
                    scale = scale + fabs(d[k]);
                }
                if (scale == 0.0) {
                    e[i] = d[i-1];
                    for (int j = 0; j < i; j++) {
                        d[j] = V[i-1][j];
                        V[i][j] = 0.0;
                        V[j][i] = 0.0;
                    }
                } else {
                    
                    // Generate Householder vector.
                    
                    for (int k = 0; k < i; k++) {
                        d[k] /= scale;
                        h += d[k] * d[k];
                    }
                    double f = d[i-1];
                    double g = sqrt(h);
                    if (f > 0) {
                        g = -g;
                    }
                    e[i] = scale * g;
                    h = h - f * g;
                    d[i-1] = f - g;
                    for (int j = 0; j < i; j++) {
                        e[j] = 0.0;
                    }
                    
                    // Apply similarity transformation to remaining columns.
                    
                    for (int j = 0; j < i; j++) {
                        f = d[j];
                        V[j][i] = f;
                        g = e[j] + V[j][j] * f;
                        for (int k = j+1; k <= i-1; k++) {
                            g += V[k][j] * d[k];
                            e[k] += V[k][j] * f;
                        }
                        e[j] = g;
                    }
                    f = 0.0;
                    for (int j = 0; j < i; j++) {
                        e[j] /= h;
                        f += e[j] * d[j];
                    }
                    double hh = f / (h + h);
                    for (int j = 0; j < i; j++) {
                        e[j] -= hh * d[j];
                    }
                    for (int j = 0; j < i; j++) {
                        f = d[j];
                        g = e[j];
                        for (int k = j; k <= i-1; k++) {
                            V[k][j] -= (f * e[k] + g * d[k]);
                        }
                        d[j] = V[i-1][j];
                        V[i][j] = 0.0;
                    }
                }
                d[i] = h;
            }
            
            // Accumulate transformations.
            
            for (int i = 0; i < n-1; i++) {
                V[n-1][i] = V[i][i];
                V[i][i] = 1.0;
                double h = d[i+1];
                if (h != 0.0) {
                    for (int k = 0; k <= i; k++) {
                        d[k] = V[k][i+1] / h;
                    }
                    for (int j = 0; j <= i; j++) {
                        double g = 0.0;
                        for (int k = 0; k <= i; k++) {
                            g += V[k][i+1] * V[k][j];
                        }
                        for (int k = 0; k <= i; k++) {
                            V[k][j] -= g * d[k];
                        }
                    }
                }
                for (int k = 0; k <= i; k++) {
                    V[k][i+1] = 0.0;
                }
            }
            for (int j = 0; j < n; j++) {
                d[j] = V[n-1][j];
                V[n-1][j] = 0.0;
            }
            V[n-1][n-1] = 1.0;
            e[0] = 0.0;
        }
        
        // Symmetric tridiagonal QL algorithm.
        
        __host__ __device__
        void mytql2(double V[3][3], double d[3], double e[3]) {
            
            const int n=3;
            //  This is derived from the Algol procedures mytql2, by
            //  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
            //  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
            //  Fortran subroutine in EISPACK.
            
            for (int i = 1; i < n; i++) {
                e[i-1] = e[i];
            }
            e[n-1] = 0.0;
            
            double f = 0.0;
            double tst1 = 0.0;
            double eps = pow(2.0,-52.0);
            for (int l = 0; l < n; l++) {
                
                // Find small subdiagonal element
                
                tst1 = MAX(tst1,fabs(d[l]) + fabs(e[l]));
                int m = l;
                while (m < n) {
                    if (fabs(e[m]) <= eps*tst1) {
                        break;
                    }
                    m++;
                }
                
                // If m == l, d[l] is an eigenvalue,
                // otherwise, iterate.
                
                if (m > l) {
                    int iter = 0;
                    do {
                        iter = iter + 1;  // (Could check iteration count here.)
                        
                        // Compute implicit shift
                        
                        double g = d[l];
                        double p = (d[l+1] - g) / (2.0 * e[l]);
                        double r = myhypot2(p,1.0);
                        if (p < 0) {
                            r = -r;
                        }
                        d[l] = e[l] / (p + r);
                        d[l+1] = e[l] * (p + r);
                        double dl1 = d[l+1];
                        double h = g - d[l];
                        for (int i = l+2; i < n; i++) {
                            d[i] -= h;
                        }
                        f = f + h;
                        
                        // Implicit QL transformation.
                        
                        p = d[m];
                        double c = 1.0;
                        double c2 = c;
                        double c3 = c;
                        double el1 = e[l+1];
                        double s = 0.0;
                        double s2 = 0.0;
                        for (int i = m-1; i >= l; i--) {
                            c3 = c2;
                            c2 = c;
                            s2 = s;
                            g = c * e[i];
                            h = c * p;
                            r = myhypot2(p,e[i]);
                            e[i+1] = s * r;
                            s = e[i] / r;
                            c = p / r;
                            p = c * d[i] - s * g;
                            d[i+1] = h + s * (c * g + s * d[i]);
                            
                            // Accumulate transformation.
                            
                            for (int k = 0; k < n; k++) {
                                h = V[k][i+1];
                                V[k][i+1] = s * V[k][i] + c * h;
                                V[k][i] = c * V[k][i] - s * h;
                            }
                        }
                        p = -s * s2 * c3 * el1 * e[l] / dl1;
                        e[l] = s * p;
                        d[l] = c * p;
                        
                        // Check for convergence.
                        
                    } while (fabs(e[l]) > eps*tst1);
                }
                d[l] = d[l] + f;
                e[l] = 0.0;
            }
            
            // Sort eigenvalues and corresponding vectors.
            
            for (int i = 0; i < n-1; i++) {
                int k = i;
                double p = d[i];
                for (int j = i+1; j < n; j++) {
                    if (d[j] < p) {
                        k = j;
                        p = d[j];
                    }
                }
                if (k != i) {
                    d[k] = d[i];
                    d[i] = p;
                    for (int j = 0; j < n; j++) {
                        p = V[j][i];
                        V[j][i] = V[j][k];
                        V[j][k] = p;
                    }
                }
            }
        }
        
        __host__ __device__
        void eigen_decomposition(double A[3][3], double V[3][3], double d[3]) {
            const int n=3;
            double e[n];
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    V[i][j] = A[i][j];
                }
            }
            mytred2(V, d, e);
            mytql2(V, d, e);
        }
        
        

        inline __host__ __device__
        void operator()(
                        Vector<T>& res,
                        const Vector<const T>& arg,
                        const Vector<const T>& tau_diag,
                        T tau_scal,
                        bool invert_tau)
        {
            // compute A^T * A
            //Eigen::Matrix3f ata;
            double A[3][3];
            A[0][0] = arg[0];
            A[1][0] = arg[1];
            A[2][0] = arg[2];
            A[0][1] = arg[1];
            A[1][1] = arg[3];
            A[2][1] = arg[4];
            A[0][2] = arg[2];
            A[1][2] = arg[4];
            A[2][2] = arg[5];
            
            double V[3][3];
            double eig[3];
            
            //dsyevv3(A, V, eig);
            eigen_decomposition(A, V, eig);
            
            eig[0] = eig[0] < 0 ? 0.0 : eig[0];
            eig[1] = eig[1] < 0 ? 0.0 : eig[1];
            eig[2] = eig[2] < 0 ? 0.0 : eig[2];
            
            // compute T = V * S * V^T
            double t11 = V[0][0] * V[0][0] * eig[0] + V[0][1] * V[0][1] * eig[1] + V[0][2] * V[0][2] * eig[2];
            //double t12 = V[0][0] * V[1][0] * eig[0] + V[0][1] * V[1][1] * eig[1] + V[0][2] * V[1][2] * eig[2];
            //double t13 = V[0][0] * V[2][0] * eig[0] + V[0][1] * V[2][1] * eig[1] + V[0][2] * V[2][2] * eig[2];
            
            double t21 = V[1][0] * V[0][0] * eig[0] + V[1][1] * V[0][1] * eig[1] + V[1][2] * V[0][2] * eig[2];
            double t22 = V[1][0] * V[1][0] * eig[0] + V[1][1] * V[1][1] * eig[1] + V[1][2] * V[1][2] * eig[2];
            //double t23 = V[1][0] * V[2][0] * eig[0] + V[1][1] * V[2][1] * eig[1] + V[1][2] * V[2][2] * eig[2];
            
            double t31 = V[2][0] * V[0][0] * eig[0] + V[2][1] * V[0][1] * eig[1] + V[2][2] * V[0][2] * eig[2];
            double t32 = V[2][0] * V[1][0] * eig[0] + V[2][1] * V[1][1] * eig[1] + V[2][2] * V[1][2] * eig[2];
            double t33 = V[2][0] * V[2][0] * eig[0] + V[2][1] * V[2][1] * eig[1] + V[2][2] * V[2][2] * eig[2];
            
            res[0] = t11;
            res[1] = t21;
            res[2] = t31;
            res[3] = t22;
            res[4] = t32;
            res[5] = t33;
        }
    };
    
} // namespace prost

#endif // PROST_ELEM_OPERATION_IND_PSD_CONE_3X3_HPP_
