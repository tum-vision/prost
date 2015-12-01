#include "prox_epi_huber.hpp"

ProxEpiHuber::ProxEpiHuber(
  size_t index,
  size_t count,
  size_t dim,
  const std::vector<T>& g,
  T alpha,
  T label_dist)

  : Prox<T>(index, count, dim, false, false), g_(g), alpha_(alpha), label_dist_(label_dist)
{
}

ProxEpiHuber::~ProxEpiHuber()
{
  Release();
}

bool 
ProxEpiHuber::Init()
{
  return true;
}

void 
ProxEpiHuber::Release()
{
}
  
void 
ProxEpiHuber::EvalLocal(T *d_arg, T *d_res, T *d_tau, T tau, bool invert_tau)
{
}
