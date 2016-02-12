#include "factory.hpp"

#include "prost/prox/prox_elem_operation.hpp"
#include "prost/prox/prox_ind_epi_polyhedral_1d.hpp"
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

ProxIndEpiPolyhedral1D<real>*
CreateProxIndEpiPolyhedral1D(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  size_t count = (size_t) mxGetScalar(mxGetCell(data, 0));
  size_t dim = (size_t) mxGetScalar(mxGetCell(data, 1));
  bool interleaved = (bool) mxGetScalar(mxGetCell(data, 2));

  std::array<std::vector<real>, 4> coeffs_xyab;
  std::array<std::vector<size_t>, 2> coeffs_ci;

  for(int i = 0; i < 4; i++)
  {
    const mwSize *dims = mxGetDimensions(mxGetCell(data, 3 + i));
    double *val = mxGetPr(mxGetCell(data, 3 + i));
    coeffs_xyab[i] = std::vector<real>(val, val + dims[0]);
  }

  for(int i = 0; i < 2; i++)
  {
    const mwSize *dims = mxGetDimensions(mxGetCell(data, 7 + i));
    double *val = mxGetPr(mxGetCell(data, 7 + i));
    coeffs_ci[i] = std::vector<size_t>(val, val + dims[0]);
  }
  
  return new ProxIndEpiPolyhedral1D<real>(idx, count, interleaved,
					  coeffs_xyab[0],
					  coeffs_xyab[1],
					  coeffs_xyab[2],
					  coeffs_xyab[3],
					  coeffs_ci[0],
					  coeffs_ci[1]);
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
  { "ind_epi_conjquad",      CreateProxElemOperationIndEpiConjQuad },
  { "ind_epi_polyhedral_1d", CreateProxIndEpiPolyhedral1D          },

  // The end.
  { "END",                   nullptr                               },
};

BlockRegistry custom_block_reg[] = 
{
  { "dataterm_sublabel", CreateBlockDatatermSublabel },

  // The end.
  { "END",               nullptr                     },
};

}
