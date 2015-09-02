#include "prox_epi_conjquadr.hpp"

ProxEpiConjQuadr::ProxEpiConjQuadr(
    int index,
    int count,
    bool interleaved,
    const EpiConjQuadrCoeffs& coeffs)
    
    : Prox(index, count, 2, interleaved, false), coeffs_(coeffs)
{
  // TODO: implement me
}

ProxEpiConjQuadr::~ProxEpiConjQuadr() {
  // TODO: implement me
}

void ProxEpiConjQuadr::Evaluate(
    real *d_arg,
    real *d_result,
    real tau,
    real *d_tau,
    bool invert_step) {

  // TODO: implement me
}
