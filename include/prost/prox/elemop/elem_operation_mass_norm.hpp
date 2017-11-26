#ifndef PROST_ELEM_OPERATION_MASS_COMASS_HPP_
#define PROST_ELEM_OPERATION_MASS_COMASS_HPP_

#include "prost/prox/elemop/elem_operation.hpp"
#include "prost/prox/elemop/function_1d.hpp"
#include "prost/prox/helper.hpp"

namespace prost {

///
/// \brief Provides proximal operator of the mass norm for 2-vectors in R^4 
///
template<typename T, bool conjugate>
struct ElemOperationMass4 : public ElemOperation<6, 0>
{
  __host__ __device__
  ElemOperationMass4(size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) { }

  inline __host__ __device__
  void operator()(
    Vector<T>& res,
    const Vector<const T>& arg,
    const Vector<const T>& tau_diag,
    T tau_scal,
    bool invert_tau)
  {
    const double tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);
    
    double M[16], Q[16], D[16];
    double U[4], S[2], V[4], FinalU[16];

    // TODO: implement multiplications with P and P^T more efficiently 
    // this is simply swapping of rows/cols
    const double P[16] = { 1, 0, 0, 0,
                           0, 0, 1, 0,
                           0, 1, 0, 0,
                           0, 0, 0, 1 };

    // construct skew symmetric matrix
    M[ 0] = 0;       M[ 4] = arg[0];  M[ 8] = arg[1];  M[12] = arg[2];
    M[ 1] = -arg[0]; M[ 5] = 0;       M[ 9] = arg[3];  M[13] = arg[4];
    M[ 2] = -arg[1]; M[ 6] = -arg[3]; M[10] = 0;       M[14] = arg[5];
    M[ 3] = -arg[2]; M[ 7] = -arg[4]; M[11] = -arg[5]; M[15] = 0;

    // bring into tri-diagonal form
    helper::skewReduce4<double>(M, Q, D);

    // perform SVD on reduced 2x2 matrix
    double J[4] = { D[4], 0, D[6], D[14] };
    helper::computeSVD2x2<double>(J, U, S, V);

    // build U matrix
    double Blk[16] = { V[0], V[1], 0, 0,
		       V[2], V[3], 0, 0,
		       0, 0, U[0], U[1],
		       0, 0, U[2], U[3] };
    
    // TODO: the first two can be trivially avoided
    helper::matMult4_BT<double>(FinalU, Blk, P);
    helper::matMult4<double>(Blk, P, FinalU);
    helper::matMult4_AT<double>(FinalU, Q, Blk);

    if(conjugate) {
      S[0] = fmin(fmax(S[0], (double)-1), (double)1);
      S[1] = fmin(fmax(S[1], (double)-1), (double)1);
    }
    else {
      Function1DAbs<double> shrink;
      S[0] = shrink(S[0], tau, 0., 0.);
      S[1] = shrink(S[1], tau, 0., 0.);
    }

    // reconstruct result 
    double Xi[16] = { 0, -S[0], 0, 0,
		      S[0], 0, 0, 0,
		      0, 0, 0, S[1],
		      0, 0, -S[1], 0 };

    helper::matMult4_BT<double>(Blk, Xi, FinalU);
    helper::matMult4<double>(M, FinalU, Blk);

    // convert back into 2-vector
    res[0] = M[4];
    res[1] = M[8];
    res[2] = M[12];
    res[3] = M[9];
    res[4] = M[13];
    res[5] = M[14];
  }
  
};

///
/// \brief Provides proximal operator of the mass for 2-vectors in R^5
///
template<typename T, bool conjugate>
struct ElemOperationMass5 : public ElemOperation<10, 0>
{
  __host__ __device__
  ElemOperationMass5(size_t dim, SharedMem<SharedMemType, GetSharedMemCount>& shared_mem) { }

  inline __host__ __device__
  void operator()(
    Vector<T>& res,
    const Vector<const T>& arg,
    const Vector<const T>& tau_diag,
    T tau_scal,
    bool invert_tau)
  {
    const double tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);

    double M[25], Q[25], D[25];
    double U[4], S[2], V[4];
    double Givens1[25], Givens2[25];

    // construct skew symmetric matrix
    M[ 0] = 0;       M[ 5] = arg[0];  M[10] = arg[1];  M[15] = arg[2];  M[20] = arg[3];
    M[ 1] = -arg[0]; M[ 6] = 0;       M[11] = arg[4];  M[16] = arg[5];  M[21] = arg[6];
    M[ 2] = -arg[1]; M[ 7] = -arg[4]; M[12] = 0;       M[17] = arg[7];  M[22] = arg[8];
    M[ 3] = -arg[2]; M[ 8] = -arg[5]; M[13] = -arg[7]; M[18] = 0;       M[23] = arg[9];
    M[ 4] = -arg[3]; M[ 9] = -arg[6]; M[14] = -arg[8]; M[19] = -arg[9]; M[24] = 0;

    // bring into tri-diagonal form
    helper::skewReduce5<double>(M, Q, D);    

    // reduce to 4x4 case using Givens rotations
    helper::givens5<double>(D, 4, 2, 3, Givens1);
    helper::givens5<double>(D, 4, 0, 1, Givens2);

    // perform SVD on reduced 2x2 matrix
    double J[4] = { D[5], 0, D[7], D[17] };
    helper::computeSVD2x2<double>(J, U, S, V);
    
    // compute U matrix
    double Blk[25] = { V[0], 0, V[1], 0, 0,
                       0, U[0], 0, U[1], 0,
                       V[2], 0, V[3], 0, 0,
                       0, U[2], 0, U[3], 0, 
                       0, 0, 0, 0, 1 };

    // Input = M * Xi * M';
    helper::matMultn<double, 5>(M, Givens2, Blk);
    helper::matMultn<double, 5>(Blk, Givens1, M);
    helper::matMultn_AT<double, 5>(M, Q, Blk);
    
    if(conjugate) {
      S[0] = fmin(fmax(S[0], (double)-1), (double)1);
      S[1] = fmin(fmax(S[1], (double)-1), (double)1);
    }
    else {
      Function1DAbs<double> shrink;
      S[0] = shrink(S[0], tau, 0., 0.);
      S[1] = shrink(S[1], tau, 0., 0.);
    }

    // reconstruct result
    double Xi[25] = { 0, -S[0], 0, 0, 0, 
                      S[0], 0, 0, 0, 0, 
                      0, 0, 0, S[1], 0,  
                      0, 0, -S[1], 0, 0,
                      0, 0, 0, 0, 0 };

    helper::matMultn_BT<double, 5>(Blk, Xi, M); // TODO: optimize this
    helper::matMultn<double, 5>(Q, M, Blk);

    // convert back into 2-vector
    res[0] = Q[5];
    res[1] = Q[10];
    res[2] = Q[15];
    res[3] = Q[20];
    res[4] = Q[11];
    res[5] = Q[16];
    res[6] = Q[21];
    res[7] = Q[17];
    res[8] = Q[22];
    res[9] = Q[23];
  }

};

}

#endif
