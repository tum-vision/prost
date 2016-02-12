#include "factory.hpp"

#include "prost/prox/prox_elem_operation.hpp"
#include "prost/prox/elemop/elem_operation_ind_epi_conjquad.hpp"
#include "prost/linop/block_dataterm_sublabel.hpp"

namespace mex
{

using namespace prost;

ProxElemOperation<real, ElemOperationIndEpiConjQuad<real> >*
CreateProxElemOperationIndEpiConjQuad(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  std::array<std::vector<real>, 5> coeffs;
  
  for(int i = 0; i < 5; i++) {
    const mwSize *dims = mxGetDimensions(mxGetCell(data, 3 + i));
    double *val = mxGetPr(mxGetCell(data, 3 + i));
    coeffs[i] = std::vector<real>(val, val + dims[0]);
  }

  return new ProxElemOperation<real, ElemOperationIndEpiConjQuad<real> >(
    idx, count, dim, interleaved, diagsteps, coeffs);
}

BlockDatatermSublabel<real>*
CreateBlockDatatermSublabel(size_t row, size_t col, const mxArray *data)
{
  size_t nx = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t ny = (size_t) mxGetScalar(mxGetCell(data, 1));
  size_t L = (size_t) mxGetScalar(mxGetCell(data, 2));
  real left = (real) mxGetScalar(mxGetCell(data, 3));
  real right = (real) mxGetScalar(mxGetCell(data, 4));

  return new BlockDatatermSublabel<real>(row, col, nx, ny, L, left, right);    
}

ProxRegistry custom_prox_reg[] = 
{
  { "elem_operation:ind_epi_conjquad", CreateProxElemOperationIndEpiConjQuad },

  // The end.
  { "END",                             nullptr                               },
};

BlockRegistry custom_block_reg[] = 
{
  { "dataterm_sublabel", CreateBlockDatatermSublabel },

  // The end.
  { "END",               nullptr                     },
};

}
