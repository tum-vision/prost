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
    const T tau = invert_tau ? (1. / (tau_scal * tau_diag[0])) : (tau_scal * tau_diag[0]);
    
    T M[16], Q[16], D[16];
    T U[4], S[2], V[4], FinalU[16];
    const T P[16] = { 1, 0, 0, 0,
			   0, 0, 1, 0,
			   0, 1, 0, 0,
			   0, 0, 0, 1 };

    // construct skew symmetric matrix
    M[ 0] = 0;       M[ 4] = arg[0];  M[ 8] = arg[1];  M[12] = arg[2];
    M[ 1] = -arg[0]; M[ 5] = 0;       M[ 9] = arg[3];  M[13] = arg[4];
    M[ 2] = -arg[1]; M[ 6] = -arg[3]; M[10] = 0;       M[14] = arg[5];
    M[ 3] = -arg[2]; M[ 7] = -arg[4]; M[11] = -arg[5]; M[15] = 0;

    // bring into tri-diagonal form
    helper::skewReduce4<T>(M, Q, D);

    // perform SVD on reduced 2x2 matrix
    T J[4] = { D[4], 0, D[6], D[14] };
    helper::computeSVD2x2<T>(J, U, S, V);

    // build U matrix
    T Blk[16] = { V[0], V[1], 0, 0,
		       V[2], V[3], 0, 0,
		       0, 0, U[0], U[1],
		       0, 0, U[2], U[3] };
    
    helper::matMult4_BT<T>(FinalU, Blk, P);
    helper::matMult4<T>(Blk, P, FinalU);
    helper::matMult4_AT<T>(FinalU, Q, Blk);

    if(conjugate) {
      S[0] = min(max(S[0], static_cast<T>(-1.)), static_cast<T>(1.));
      S[1] = min(max(S[1], static_cast<T>(-1.)), static_cast<T>(1.));
    }
    else {
      const Function1DAbs<T> shrink;
      S[0] = shrink(S[0], tau, 0., 0.);
      S[1] = shrink(S[1], tau, 0., 0.);
    }

    // reconstruct result 
    T Xi[16] = { 0, -S[0], 0, 0,
		      S[0], 0, 0, 0,
		      0, 0, 0, S[1],
		      0, 0, -S[1], 0 };

    helper::matMult4_BT<T>(Blk, Xi, FinalU);
    helper::matMult4<T>(M, FinalU, Blk);

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
    // TODO
    
    if(conjugate) {
    }
    else {
    }
  }

};

}

#endif
