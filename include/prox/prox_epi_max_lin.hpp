#ifndef PROX_EPI_MAX_LIN_HPP_
#define PROX_EPI_MAX_LIN_HPP_

#include <vector>

#include "prox.hpp"

template<typename T>
struct EpiMaxLinCoeffs {
    std::vector<T> t, b;
    std::vector<size_t> count, index;    
};

template<typename T>
struct EpiMaxLinCoeffsDevice {
    T *d_ptr_b;
    T *d_ptr_t;
    size_t *d_ptr_count;
    size_t *d_ptr_index;
};
/**
 * @brief ...
 */
template<typename T>
class ProxEpiMaxLin : public Prox<T> {
 public:
  ProxEpiMaxLin(size_t index,
                size_t count,
                size_t dim,
                bool interleaved,
                const EpiMaxLinCoeffs<T>& coeffs);

  virtual ~ProxEpiMaxLin();

  virtual size_t gpu_mem_amount(); 
  
  virtual bool Init();
  virtual void Release();
  
protected:
  virtual void EvalLocal(T *d_arg,
                         T *d_res,
                         T *d_tau,
                         T tau,
                         bool invert_tau);
  
  EpiMaxLinCoeffs<T> coeffs_;
  EpiMaxLinCoeffsDevice<T> coeffs_dev_;
  size_t max_count_;
};

#ifdef __CUDACC__

#define MIN(x,y) ( (x) < (y) ? (x) : (y) )
#define MAX(x,y) ((x)>(y)?(x):(y))
#define SIGN(a, b) ((b) >= 0.0 ? fabs(a) : -fabs(a))

template<typename T>
inline __device__ T calcEuclDist(T* v1, T* v2, size_t dim) {
    T d = 0.0;
    for(size_t i = 0; i < dim; i++) {
        d += pow(v1[i]-v2[i], static_cast<T>(2.0));
    }
    return sqrt(d);
}

template<typename T>
inline __device__ bool isFeasible(T* t, T* b, T* v, size_t m, size_t n) {
    for(size_t j = 0; j < n; j++) {
      T val = 0;
      for(size_t i = 0; i < m; i++) {
        val += t[i*n + j]*v[i];
      }
      if(val-1e-3 > -b[j]) {
        return false;
      }
    }
    return true;
}

// proj = arg min 0.5*||x-x_bar||^2 s.t. A^T x = -q
template<typename T>
inline __device__ int projSubspace(T* a, T* u, T* w, T*v, T* pinv, size_t m, size_t n, size_t dim, T* q, T* x, T* proj) {
    // copy a to u since a must not be overwritten...
    for(size_t i = 0; i < m; i++) {
        for(size_t j = 0; j < n; j++) {
            u[i*dim + j] = a[i*dim + j];
        }
    }
    
    int res = calcPseudoInv(u, w, v, proj, m, n, dim, pinv);
    
    for(size_t i = 0; i < dim; i++) {
      for(size_t j = 0; j < dim; j++) {
        u[i*dim + j] = 0;
        for(size_t k = 0; k < n; k++) {
          u[i*dim + j] -= a[i*dim + k]*pinv[k* dim + j];
        }
      }
      u[i*dim + i] += 1;
    }

    for(size_t i = 0; i < dim; i++) {
      proj[i] = 0;
      for(size_t j = 0; j < n; j++) {
        proj[i] -= pinv[j* dim + i]*q[j];
      }
    }
    
    // write out result
    for(size_t i=0; i < dim; i++) {
      for(size_t j = 0; j < dim; j++) {
        proj[i] += u[i*dim + j]*x[j];
      }
    }
    
    return res;
}

template<typename T>
inline __device__ int calcPseudoInv(T* a, T* w, T* v, T* rv1, size_t m, size_t n, size_t dim, T* pinv) {
    int res = decompSVD(a, m, n, dim, w, v, rv1);
    
    for(size_t j = 0; j < n; j++) {
        for(size_t i = 0; i < m; i++) {
            pinv[j*dim + i] = 0;
            for(size_t k = 0; k < n; k++) {
                if(w[k] > 1e-5)
                    pinv[j*dim + i] += v[j*dim + k]*(1/w[k])*a[i* dim + k];
            }
        }
    }
    
    return res;
}

template<typename T>
inline __device__ T calcPythag(T a, T b) {
    T at = fabs(a), bt = fabs(b), ct, result;

    if (at > bt) {
        ct = bt / at;
        result = at * sqrt(1.0 + ct * ct);
    } else if (bt > 0.0) {
        ct = at / bt;
        result = bt * sqrt(1.0 + ct * ct);
    } else
        result = 0.0;
    
    return(result);
}

/**
 * @brief Computes the SVD of A.
 * http://www.public.iastate.edu/~dicook/JSS/paper/code/svd.c
 */

/* 
 * svdcomp - SVD decomposition routine. 
 * Takes an mxn matrix a and decomposes it into udv, where u,v are
 * left and right orthogonal transformation matrices, and d is a 
 * diagonal matrix of singular values.
 *
 * This routine is adapted from svdecomp.c in XLISP-STAT 2.1 which is 
 * code from Numerical Recipes adapted by Luke Tierney and David Betz.
 *
 * Input to dsvd is as follows:
 *   a = mxn matrix to be decomposed, gets overwritten with u
 *   m = row dimension of a
 *   n = column dimension of a
 *   w = returns the vector of singular values of a
 *   v = returns the right orthogonal transformation matrix
*/
template<typename T>
inline __device__ int decompSVD(T* a, size_t m, size_t n, size_t dim, T* w, T* v, T* rv1) {
    int flag, i, its, j, jj, k, l, nm;
    T c, f, h, s, x, y, z;
    T anorm = 0.0, g = 0.0, scale = 0.0;
  
    if (m < n) {
        // fprintf(stderr, "#rows must be >= #cols \n");
        return(0);
    }
  
    /* Householder reduction to bidiagonal form */
    for (i = 0; i < n; i++) {
        /* left-hand reduction */
        l = i + 1;
        rv1[i] = scale * g;
        g = s = scale = 0.0;
        if (i < m) {
            for (k = i; k < m; k++) 
                scale += fabs(a[k*dim+i]);
            
            if (scale) {
                for (k = i; k < m; k++) {
                    a[k*dim+i] = a[k*dim+i] / scale;
                    s += (a[k*dim+i] * a[k*dim+i]);
                }
                
                f = a[i*dim+i];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i*dim+i] = f - g;
                if (i != n - 1) {
                    for (j = l; j < n; j++) {
                        for (s = 0.0, k = i; k < m; k++) 
                            s += (a[k*dim+i] * a[k*dim+j]);
                        f = s / h;
                        for (k = i; k < m; k++) 
                            a[k*dim+j] += (f * a[k*dim+i]);
                    }
                }
                for (k = i; k < m; k++) 
                    a[k*dim+i] = a[k*dim+i]*scale;
            }
        }
        w[i] = scale * g;
    
        /* right-hand reduction */
        g = s = scale = 0.0;
        if (i < m && i != n - 1) {
            for (k = l; k < n; k++) 
                scale += fabs(a[i*dim+k]);
            if (scale) {
                for (k = l; k < n; k++) {
                    a[i*dim+k] = a[i*dim+k]/scale;
                    s += (a[i*dim+k] * a[i*dim+k]);
                }
                f = a[i*dim+l];
                g = -SIGN(sqrt(s), f);
                h = f * g - s;
                a[i*dim+l] = f - g;
                for (k = l; k < n; k++) 
                    rv1[k] = a[i*dim+k] / h;
                
                if (i != m - 1) {
                    for (j = l; j < m; j++) {
                        for (s = 0.0, k = l; k < n; k++) 
                            s += (a[j*dim+k] * a[i*dim+k]);
                        
                        for (k = l; k < n; k++) 
                            a[j*dim+k] += (s * rv1[k]);
                    }
                }
                for (k = l; k < n; k++) 
                    a[i*dim+k] = a[i*dim+k]*scale;
            }
        }
        anorm = MAX(anorm, (fabs(w[i]) + fabs(rv1[i])));
    }
  
    /* accumulate the right-hand transformation */
    for (i = n - 1; i >= 0; i--) {
        if (i < n - 1) {
            if (g) {
                for (j = l; j < n; j++)
                    v[j*dim+i] = ((a[i*dim+j] / a[i*dim+l]) / g);
                    /* T division to avoid underflow */
                for (j = l; j < n; j++) {
                    for (s = 0.0, k = l; k < n; k++) 
                        s += (a[i*dim+k] * v[k*dim+j]);
                    
                    for (k = l; k < n; k++) 
                        v[k*dim+j] += (s * v[k*dim+i]);
                }
            }
            for (j = l; j < n; j++) 
                v[i*dim+j] = v[j*dim+i] = 0.0;
        }
        v[i*dim+i] = 1.0;
        g = rv1[i];
        l = i;
    }
  
    /* accumulate the left-hand transformation */
    for (i = n - 1; i >= 0; i--) {
        l = i + 1;
        g = w[i];
        if (i < n - 1) 
            for (j = l; j < n; j++) 
                a[i*dim+j] = 0.0;
        if (g) {
            g = 1.0 / g;
            if (i != n - 1) {
                for (j = l; j < n; j++) {
                    for (s = 0.0, k = l; k < m; k++) 
                        s += (a[k*dim+i] * a[k*dim+j]);
                    f = (s / a[i*dim+i]) * g;
                    for (k = i; k < m; k++) 
                        a[k*dim+j] += (f * a[k*dim+i]);
                }
            }
            for (j = i; j < m; j++) 
                a[j*dim+i] = a[j*dim+i]*g;
        }
        else {
            for (j = i; j < m; j++) 
                a[j*dim+i] = 0.0;
        }
        ++a[i*dim+i];
    }

    /* diagonalize the bidiagonal form */
    for (k = n - 1; k >= 0; k--) {
        /* loop over singular values */
        for (its = 0; its < 30; its++) {
            /* loop over allowed iterations */
            flag = 1;
            for (l = k; l >= 0; l--) {
                /* test for splitting */
                nm = l - 1;
                
                if (fabs(rv1[l]) + anorm == anorm) {
                    flag = 0;
                    break;
                }
                if (fabs(w[nm]) + anorm == anorm) 
                    break;
            }
            if (flag) {
                c = 0.0;
                s = 1.0;
                for (i = l; i <= k; i++) {
                    f = s * rv1[i];
                    if (fabs(f) + anorm != anorm) {
                        g = w[i];
                        h = calcPythag(f, g);
                        w[i] = h; 
                        h = 1.0 / h;
                        c = g * h;
                        s = (- f * h);
                        for (j = 0; j < m; j++) {
                            y = a[j*dim+nm];
                            z = a[j*dim+i];
                            a[j*dim+nm] = y * c + z * s;
                            a[j*dim+i] = z * c - y * s;
                        }
                    }
                }
            }
            z = w[k];
            if (l == k) {
                /* convergence */
                if (z < 0.0) {
                    /* make singular value nonnegative */
                    w[k] = -z;
                    for (j = 0; j < n; j++) 
                        v[j*dim+k] = -v[j*dim+k];
                }
                break;
            }
            if (its >= 30) {
                // fprintf(stderr, "No convergence after 30,000! iterations \n");
                return(0);
            }
    
            /* shift from bottom 2 x 2 minor */
            x = w[l];
            nm = k - 1;
            y = w[nm];
            g = rv1[nm];
            h = rv1[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = calcPythag(f, static_cast<T>(1.0));
            f = ((x - z) * (x + z) + h * ((y / (f + SIGN(g, f))) - h)) / x;
          
            /* next QR transformation */
            c = s = 1.0;
            for (j = l; j <= nm; j++) {
                i = j + 1;
                g = rv1[i];
                y = w[i];
                h = s * g;
                g = c * g;
                z = calcPythag(f, h);
                rv1[j] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = g * c - x * s;
                h = y * s;
                y = y * c;
                for (jj = 0; jj < n; jj++) {
                    x = v[jj*dim+j];
                    z = v[jj*dim+i];
                    v[jj*dim+j] = x * c + z * s;
                    v[jj*dim+i] = z * c - x * s;
                }
                z = calcPythag(f, h);
                w[j] = z;
                if (z) {
                    z = 1.0 / z;
                    c = f * z;
                    s = h * z;
                }
                
                f = (c * g) + (s * y);
                x = (c * y) - (s * g);
                for (jj = 0; jj < m; jj++) {
                    y = a[jj*dim+j];
                    z = a[jj*dim+i];
                    a[jj*dim+j] = y * c + z * s;
                    a[jj*dim+i] = z * c - y * s;
                }
            }
            rv1[l] = 0.0;
            rv1[k] = f;
            w[k] = x;
        }
    }
    
    return(1);
}


#endif

#endif
