#include "prox_zero.hpp"

ProxZero::ProxZero(int index, int count)
    : Prox(index, count, 1, false, true)
{
}

ProxZero::~ProxZero() {
}

void ProxZero::Evaluate(
    real *d_arg,
    real *d_result,
    real tau,
    real *d_tau,
    bool invert_step)
{
  cudaMemcpy(&d_result[index_],
             &d_arg[index_],
             sizeof(real) * count_,
             cudaMemcpyDeviceToDevice);
}
