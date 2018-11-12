#include "factory.hpp"

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

using namespace prost;

namespace matlab
{

mxArray *Solver_interm_cb_handle = nullptr;

static map<string, function<Prox<real>*(size_t, size_t, bool, const mxArray*)>> default_prox_reg = {
  { "elem_operation:ind_simplex",                     CreateProxElemOperationIndSimplex                                                     },
  { "elem_operation:ind_sum",                         CreateProxElemOperationIndSum                                                         },
  { "elem_operation:ind_psd_cone_3x3",                CreateProxElemOperationIndPsdCone3x3                                                  },
  { "elem_operation:1d:zero",                         CreateProxElemOperation1D<Function1DZero<real>>                                       },
  { "elem_operation:1d:abs",                          CreateProxElemOperation1D<Function1DAbs<real>>                                        },
  { "elem_operation:1d:square",                       CreateProxElemOperation1D<Function1DSquare<real>>                                     },
  { "elem_operation:1d:ind_leq0",                     CreateProxElemOperation1D<Function1DIndLeq0<real>>                                    },
  { "elem_operation:1d:ind_geq0",                     CreateProxElemOperation1D<Function1DIndGeq0<real>>                                    },
  { "elem_operation:1d:ind_eq0",                      CreateProxElemOperation1D<Function1DIndEq0<real>>                                     },
  { "elem_operation:1d:ind_box01",                    CreateProxElemOperation1D<Function1DIndBox01<real>>                                   },
  { "elem_operation:1d:max_pos0",                     CreateProxElemOperation1D<Function1DMaxPos0<real>>                                    },
  { "elem_operation:1d:l0",                           CreateProxElemOperation1D<Function1DL0<real>>                                         },
  { "elem_operation:1d:huber",                        CreateProxElemOperation1D<Function1DHuber<real>>                                      },
  { "elem_operation:1d:lq",                           CreateProxElemOperation1D<Function1DLq<real>>                                         },
  { "elem_operation:1d:lq_plus_eps",                  CreateProxElemOperation1D<Function1DLqPlusEps<real>>                                  },
  { "elem_operation:1d:trunclin",                     CreateProxElemOperation1D<Function1DTruncLinear<real>>                                },
  { "elem_operation:1d:truncquad",                    CreateProxElemOperation1D<Function1DTruncQuad<real>>                                  },
  { "elem_operation:norm2:zero",                      CreateProxElemOperationNorm2<Function1DZero<real>>                                    },
  { "elem_operation:norm2:abs",                       CreateProxElemOperationNorm2<Function1DAbs<real>>                                     },
  { "elem_operation:norm2:square",                    CreateProxElemOperationNorm2<Function1DSquare<real>>                                  },
  { "elem_operation:norm2:ind_leq0",                  CreateProxElemOperationNorm2<Function1DIndLeq0<real>>                                 },
  { "elem_operation:norm2:ind_geq0",                  CreateProxElemOperationNorm2<Function1DIndGeq0<real>>                                 },
  { "elem_operation:norm2:ind_eq0",                   CreateProxElemOperationNorm2<Function1DIndEq0<real>>                                  },
  { "elem_operation:norm2:ind_box01",                 CreateProxElemOperationNorm2<Function1DIndBox01<real>>                                },
  { "elem_operation:norm2:max_pos0",                  CreateProxElemOperationNorm2<Function1DMaxPos0<real>>                                 },
  { "elem_operation:norm2:l0",                        CreateProxElemOperationNorm2<Function1DL0<real>>                                      },
  { "elem_operation:norm2:huber",                     CreateProxElemOperationNorm2<Function1DHuber<real>>                                   },
  { "elem_operation:norm2:lq",                        CreateProxElemOperationNorm2<Function1DLq<real>>                                      },
  { "elem_operation:norm2:lq_plus_eps",               CreateProxElemOperationNorm2<Function1DLqPlusEps<real>>                               },
  { "elem_operation:norm2:trunclin",                  CreateProxElemOperationNorm2<Function1DTruncLinear<real>>                             },
  { "elem_operation:norm2:truncquad",                 CreateProxElemOperationNorm2<Function1DTruncQuad<real>>                               },
  { "elem_operation:singular_nx2:sum_1d:zero",        CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DZero<real>>>       },
  { "elem_operation:singular_nx2:sum_1d:abs",         CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DAbs<real>>>        },
  { "elem_operation:singular_nx2:sum_1d:square",      CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DSquare<real>>>     },
  { "elem_operation:singular_nx2:sum_1d:ind_leq0",    CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DIndLeq0<real>>>    },
  { "elem_operation:singular_nx2:sum_1d:ind_geq0",    CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DIndGeq0<real>>>    },
  { "elem_operation:singular_nx2:sum_1d:ind_eq0",     CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DIndEq0<real>>>     },
  { "elem_operation:singular_nx2:sum_1d:ind_box01",   CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DIndBox01<real>>>   },
  { "elem_operation:singular_nx2:sum_1d:max_pos0",    CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DMaxPos0<real>>>    },
  { "elem_operation:singular_nx2:sum_1d:l0",          CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DL0<real>>>         },
  { "elem_operation:singular_nx2:sum_1d:huber",       CreateProxElemOperationSingularNx2<Function2DSum1D<real, Function1DHuber<real>>>      },
  { "elem_operation:singular_nx2:ind_l1_ball",        CreateProxElemOperationSingularNx2<Function2DIndL1Ball<real>>                         },
  { "elem_operation:singular_nx2:moreau:ind_l1_ball", CreateProxElemOperationSingularNx2<Function2DMoreau<real, Function2DIndL1Ball<real>>> },
  { "elem_operation:mass4",                           CreateProxElemOperationMass4<false>                                                   },
  { "elem_operation:ind_comass4_ball",                CreateProxElemOperationMass4<true>                                                    },
  { "elem_operation:mass5",                           CreateProxElemOperationMass5<false>                                                   },
  { "elem_operation:ind_comass5_ball",                CreateProxElemOperationMass5<true>                                                    },
  { "ind_epi_quad",                                   CreateProxIndEpiQuad                                                                  },
  { "ind_halfspace",                                  CreateProxIndHalfspace                                                                },
  { "ind_range",                                      CreateProxIndRange                                                                    },
  { "ind_soc",                                        CreateProxIndSOC                                                                      },
  { "ind_sum",                                        CreateProxIndSum                                                                      },
  { "moreau",                                         CreateProxMoreau                                                                      },
  { "permute",                                        CreateProxPermute                                                                     },
  { "transform",                                      CreateProxTransform                                                                   },
  { "zero",                                           CreateProxZero                                                                        },
};

const static map<string, function<Block<real>*(size_t, size_t, const mxArray*)>> default_block_reg = {
  { "dense",          CreateBlockDense        },
  { "diags",          CreateBlockDiags        },
  { "gradient2d",     CreateBlockGradient2D   },
  { "gradient3d",     CreateBlockGradient3D   },
  { "id_kron_dense",  CreateBlockIdKronDense  },
  { "id_kron_sparse", CreateBlockIdKronSparse },
  { "sparse",         CreateBlockSparse       },
  { "dense_kron_id",  CreateBlockDenseKronId  },
  { "sparse_kron_id", CreateBlockSparseKronId },
  { "zero",           CreateBlockZero         },
};

const static map<string, function<Backend<real>*(const mxArray*)>> default_backend_reg = {
  { "admm", CreateBackendADMM },
  { "pdhg", CreateBackendPDHG },
};

bool
SolverIntermCallback(int iter, const std::vector<real>& primal, const std::vector<real>& dual)
{
  mxArray *cb_rhs[4];
  cb_rhs[0] = Solver_interm_cb_handle;
  cb_rhs[1] = mxCreateDoubleScalar(iter);
  cb_rhs[2] = mxCreateDoubleMatrix(primal.size(), 1, mxREAL); 
  cb_rhs[3] = mxCreateDoubleMatrix(dual.size(), 1, mxREAL);

  mxArray *cb_lhs[1];

  std::copy(primal.begin(), primal.end(), mxGetPr(cb_rhs[2]));
  std::copy(dual.begin(), dual.end(), mxGetPr(cb_rhs[3]));
  
  mexCallMATLAB(1, cb_lhs, 4, cb_rhs, "feval");

  mxDestroyArray(cb_rhs[2]);
  mxDestroyArray(cb_rhs[3]);

  bool is_converged = static_cast<double>(mxGetScalar(cb_lhs[0]));

  return is_converged;
}

// Reads a vector from matlab and converts it to std::vector of
// the specified type.
template<typename T>
std::vector<T> GetVector(const mxArray *p)
{
  const mwSize *dims = mxGetDimensions(p);
  
  if(dims[1] != 1 && dims[0] != 1)
    throw Exception("Vector has to be Nx1 or 1xN.");

  if(dims[0] == 0 || dims[1] == 0)
    throw Exception("Empty vector passed.");

  if(mxIsDouble(p))
  {
    double *val = mxGetPr(p);
    
    if(dims[1] == 1)
      return std::vector<T>(val, val + dims[0]);
    else
      return std::vector<T>(val, val + dims[1]);
  }
  else if(mxIsSingle(p))
  {
    float *val = (float *)mxGetPr(p);
    
    if(dims[1] == 1)
      return std::vector<T>(val, val + dims[0]);
    else
      return std::vector<T>(val, val + dims[1]);
  }
  else
    throw Exception("Argument has to be passed as a vector of type single or double.");
}

// Reads a cell-array of matlab vectors into an std::array of std::vector.
template<size_t COEFFS_COUNT>
void GetCoefficients(
  std::array<std::vector<real>, COEFFS_COUNT>& coeffs, 
  const mxArray *cell_arr, 
  size_t count) 
{
  const mwSize *dims = mxGetDimensions(cell_arr);
  if(dims[0] * dims[1] < COEFFS_COUNT)
  {
    throw Exception("Cell array of coefficients is too small.");
  }
  
  for(size_t i = 0; i < COEFFS_COUNT; i++) {
    coeffs[i] = GetVector<real>(mxGetCell(cell_arr, i));

    if(coeffs[i].size() != 1 && coeffs[i].size() != count)
      throw Exception("Size of coefficients should be either 1 or count.");
  }
}

// Reads a cell array and writes its contents into a vector. A 2d cell array
// gets linearized.
std::vector<const mxArray*> GetCellArray(const mxArray *cell_array)
{
  if(mxGetNumberOfDimensions(cell_array) > 2)
    throw Exception("Cannot handle cell-array with dim > 2.");

  if(cell_array == nullptr)
    throw Exception("Tried to run GetCellArray on non-existing array.");
  
  const mwSize *dims = mxGetDimensions(cell_array);
  
  std::vector<const mxArray*> cells;
  for(int i = 0; i < dims[0] * dims[1]; i++)
  {
    cells.push_back(mxGetCell(cell_array, i));
  }

  return cells;
}

// Helper function for reading scalars
template<typename T>
T GetScalarFromCellArray(const mxArray *p, size_t index)
{
  const mwSize *dims = mxGetDimensions(p);

  if(index >= dims[0] * dims[1])
    throw Exception("Out-of-bounds access into cell-array.");
  
  return static_cast<T>(mxGetScalar(mxGetCell(p, index)));
}

template<>
bool GetScalarFromCellArray(const mxArray *p, size_t index)
{
  const mwSize *dims = mxGetDimensions(p);

  if(index >= dims[0] * dims[1])
    throw Exception("Out-of-bounds access into cell-array.");

  return static_cast<bool>(mxGetScalar(mxGetCell(p, index)) > 0.);
}

template<typename T>
T GetScalarFromField(const mxArray *p, const std::string& name)
{
  const mxArray *f = mxGetField(p, 0, name.c_str());
  if(f == nullptr) {
    std::stringstream ss;

    ss << "Field with name '" << name << "' not found.";
    throw Exception(ss.str());
  }
  
  return static_cast<T>(mxGetScalar(f));
}

template<>
bool GetScalarFromField(const mxArray *p, const std::string& name) {
  const mxArray *f = mxGetField(p, 0, name.c_str());
  if(f == nullptr) {
    std::stringstream ss;

    ss << "Field with name '" << name << "' not found.";
    throw Exception(ss.str());
  }
  
  return static_cast<bool>(mxGetScalar(f) > 0.);
}

ProxMoreau<real>* 
CreateProxMoreau(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  return new ProxMoreau<real>(CreateProx(mxGetCell(data, 0)));
}

ProxPermute<real>* 
CreateProxPermute(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  vector<int> perm = GetVector<int>(mxGetCell(data, 1));
  
  return new ProxPermute<real>(CreateProx(mxGetCell(data, 0)), perm);
}

ProxTransform<real>*
CreateProxTransform(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  std::array<std::vector<real>, 5> coeffs;
  GetCoefficients<5>(coeffs, data, size);

  return new ProxTransform<real>(
    CreateProx(mxGetCell(data, 5)), 
    coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4]);
}
    
ProxZero<real>* 
CreateProxZero(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  return new ProxZero<real>(idx, size);
}

template<class FUN_1D> 
ProxElemOperation<real, ElemOperation1D<real, FUN_1D> >*
CreateProxElemOperation1D(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  size_t count = GetScalarFromCellArray<size_t>(data, 0); 
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  std::array<std::vector<real>, 7> coeffs;
  GetCoefficients<7>(coeffs, mxGetCell(data, 3), size);

  return new ProxElemOperation<real, ElemOperation1D<real, FUN_1D>>(
    idx, count, dim, interleaved, diagsteps, coeffs);   
}

template<class FUN_1D>
ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D> >* 
CreateProxElemOperationNorm2(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  std::array<std::vector<real>, 7> coeffs;
  GetCoefficients<7>(coeffs, mxGetCell(data, 3), count);
  
  return new ProxElemOperation<real, ElemOperationNorm2<real, FUN_1D>>(
    idx, count, dim, interleaved, diagsteps, coeffs);   
}

template<class FUN_2D>
ProxElemOperation<real, ElemOperationSingularNx2<real, FUN_2D> >*
CreateProxElemOperationSingularNx2(size_t idx, size_t size, bool diagsteps, const mxArray *data) {
  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  std::array<std::vector<real>, 7> coeffs;
  GetCoefficients<7>(coeffs, mxGetCell(data, 3), count);
  
  return new ProxElemOperation<real, ElemOperationSingularNx2<real, FUN_2D> >(
    idx, count, dim, interleaved, diagsteps, coeffs);   
}


ProxElemOperation<real, ElemOperationIndSimplex<real> >* 
CreateProxElemOperationIndSimplex(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  return new ProxElemOperation<real, ElemOperationIndSimplex<real> >(idx, count, dim, interleaved, diagsteps);   
}

ProxElemOperation<real, ElemOperationIndSum<real> >* 
CreateProxElemOperationIndSum(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  return new ProxElemOperation<real, ElemOperationIndSum<real> >(idx, count, dim, interleaved, diagsteps);   
}
    
ProxElemOperation<real, ElemOperationIndPsdCone3x3<real> >*
CreateProxElemOperationIndPsdCone3x3(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
    size_t count = GetScalarFromCellArray<size_t>(data, 0);
    size_t dim = GetScalarFromCellArray<size_t>(data, 1);
    bool interleaved = GetScalarFromCellArray<bool>(data, 2);
        
    return new ProxElemOperation<real, ElemOperationIndPsdCone3x3<real> >(idx, count, dim, interleaved, diagsteps);
}
    
ProxIndEpiQuad<real>* 
CreateProxIndEpiQuad(size_t idx, size_t size, bool diagsteps, const mxArray *data) 
{
  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);
  const mxArray *coeffs = mxGetCell(data, 3);

  std::vector<real> a = GetVector<real>(mxGetCell(coeffs, 0));
  std::vector<real> b = GetVector<real>(mxGetCell(coeffs, 1));
  std::vector<real> c = GetVector<real>(mxGetCell(coeffs, 2));
  
  return new ProxIndEpiQuad<real>(idx, count, dim, interleaved, diagsteps, a, b, c);   
}

prost::ProxIndSOC<real>*
CreateProxIndSOC(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  real alpha = GetScalarFromCellArray<real>(data, 3);

  return new ProxIndSOC<real>(idx, count, dim, interleaved, diagsteps, alpha);
}

prost::ProxIndSum<real>*
CreateProxIndSum(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{  
  size_t dim = GetScalarFromCellArray<size_t>(data, 0);  
  std::vector<size_t> inds = GetVector<size_t>(mxGetCell(data, 1));

  size_t count = inds.size() / dim; // hacky

  //mexPrintf("CreateProxIndSum %d %d\n", count, dim);
  
  return new ProxIndSum<real>(idx, size, count, dim, inds);
}


prost::ProxIndHalfspace<real>*
CreateProxIndHalfspace(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);
  const mxArray *coeffs = mxGetCell(data, 3);

  std::vector<real> a = GetVector<real>(mxGetCell(coeffs, 0));
  std::vector<real> b = GetVector<real>(mxGetCell(coeffs, 1));
  
  return new ProxIndHalfspace<real>(idx, count, dim, interleaved, diagsteps, a, b);   
}

template<bool conjugate>
ProxElemOperation<real, ElemOperationMass4<real, conjugate> >*
CreateProxElemOperationMass4(size_t idx, size_t size, bool diagsteps, const mxArray *data) {

  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  if(dim != 6)
    throw Exception("Wrong dimension in mass norm prox");

  return new ProxElemOperation<real, ElemOperationMass4<real, conjugate>>(
    idx, count, dim, interleaved, diagsteps);
  
}
  
template<bool conjugate>
ProxElemOperation<real, ElemOperationMass5<real, conjugate> >*
CreateProxElemOperationMass5(size_t idx, size_t size, bool diagsteps, const mxArray *data) {

  size_t count = GetScalarFromCellArray<size_t>(data, 0);
  size_t dim = GetScalarFromCellArray<size_t>(data, 1);
  bool interleaved = GetScalarFromCellArray<bool>(data, 2);

  if(dim != 10)
    throw Exception("Wrong dimension in mass norm prox");

  return new ProxElemOperation<real, ElemOperationMass5<real, conjugate>>(
    idx, count, dim, interleaved, diagsteps);

}

prost::ProxIndRange<real>*
CreateProxIndRange(size_t idx, size_t size, bool diagsteps, const mxArray *data)
{
  ProxIndRange<real> *prox = new ProxIndRange<real>(idx, size, false);

  mxArray *pm = mxGetCell(data, 0);

  if(!mxIsSparse(pm))
    throw Exception("Matrix A must be sparse!");
  
  double *val = mxGetPr(pm);
  mwIndex *ind = mxGetIr(pm);
  mwIndex *ptr = mxGetJc(pm);
  const mwSize *dims = mxGetDimensions(pm);

  int nrows = dims[0];
  int ncols = dims[1];
  int nnz = ptr[ncols];

  if(size != nrows)
    throw Exception("Matrix A does not fit size of the variable!");
  
  // convert from mwIndex -> int32_t, double -> real
  std::vector<real> vec_val(val, val + nnz);
  std::vector<int32_t> vec_ptr(ptr, ptr + (ncols + 1));
  std::vector<int32_t> vec_ind(ind, ind + nnz); 
  
  prox->setA(nrows, ncols, nnz, vec_val, vec_ptr, vec_ind);

  pm = mxGetCell(data, 1);
  if(mxIsSparse(pm))
    throw Exception("Matrix AA must be dense!");
  
  nrows = mxGetM(pm);
  ncols = mxGetN(pm);
  double *vals = reinterpret_cast<double *>(mxGetData(pm));  
  std::vector<real> dense_vals(vals, vals + nrows * ncols);

  prox->setAA(nrows, ncols, dense_vals);

  return prox;
}

BlockDiags<real>*
CreateBlockDiags(size_t row, size_t col, const mxArray *pm)
{
  size_t nrows = GetScalarFromCellArray<size_t>(pm, 0);
  size_t ncols = GetScalarFromCellArray<size_t>(pm, 1);

  std::vector<real> factors = GetVector<real>(mxGetCell(pm, 2));
  std::vector<ssize_t> offsets = GetVector<ssize_t>(mxGetCell(pm, 3));

  if((factors.size() != offsets.size()) && (factors.size() != 1 || offsets.size() != 1))
    throw Exception("Mismatch of size(factors) and size(offsets).");

  size_t ndiags = factors.size();
  return new BlockDiags<real>(row, col, nrows, ncols, ndiags, offsets, factors);
}

BlockDense<real>*
CreateBlockDense(size_t row, size_t col, const mxArray *pm)
{
  size_t nrows = mxGetM(mxGetCell(pm, 0));
  size_t ncols = mxGetN(mxGetCell(pm, 0));
  double *data = reinterpret_cast<double *>(mxGetData(mxGetCell(pm, 0)));
  std::vector<real> r_data(data, data + nrows * ncols);

  //cout << nrows << "," << ncols << "," << r_data.size() << endl;
  
  return BlockDense<real>::CreateFromColFirstData(row, col, nrows, ncols, r_data);
}

prost::BlockGradient2D<real>*
CreateBlockGradient2D(size_t row, size_t col, const mxArray *pm)
{
  size_t nx = GetScalarFromCellArray<size_t>(pm, 0);
  size_t ny = GetScalarFromCellArray<size_t>(pm, 1);
  size_t L = GetScalarFromCellArray<size_t>(pm, 2);
  bool label_first = GetScalarFromCellArray<bool>(pm, 3);

  return new BlockGradient2D<real>(row, col, nx, ny, L, label_first);
}

prost::BlockGradient3D<real>*
CreateBlockGradient3D(size_t row, size_t col, const mxArray *pm)
{
  size_t nx = GetScalarFromCellArray<size_t>(pm, 0);
  size_t ny = GetScalarFromCellArray<size_t>(pm, 1);
  size_t L = GetScalarFromCellArray<size_t>(pm, 2);
  bool label_first = GetScalarFromCellArray<bool>(pm, 3);

  return new BlockGradient3D<real>(row, col, nx, ny, L, label_first);
}
  
BlockSparse<real>*
CreateBlockSparse(size_t row, size_t col, const mxArray *data)
{
  mxArray *pm = mxGetCell(data, 0);

  if(!mxIsSparse(pm))
    throw Exception("Matrix must be sparse!");
  
  double *val = mxGetPr(pm);
  mwIndex *ind = mxGetIr(pm);
  mwIndex *ptr = mxGetJc(pm);
  const mwSize *dims = mxGetDimensions(pm);

  int nrows = dims[0];
  int ncols = dims[1];
  int nnz = ptr[ncols];

  // convert from mwIndex -> int32_t, double -> real
  std::vector<real> vec_val(val, val + nnz);
  std::vector<int32_t> vec_ptr(ptr, ptr + (ncols + 1));
  std::vector<int32_t> vec_ind(ind, ind + nnz); 

  return BlockSparse<real>::CreateFromCSC(
    row, 
    col,
    nrows,
    ncols,
    nnz,
    vec_val,
    vec_ptr,
    vec_ind);
}

prost::BlockIdKronSparse<real>*
CreateBlockIdKronSparse(size_t row, size_t col, const mxArray *data)
{
  mxArray *pm = mxGetCell(data, 0);

  if(!mxIsSparse(pm))
    throw Exception("Matrix must be sparse!");

  double *val = mxGetPr(pm);
  mwIndex *ind = mxGetIr(pm);
  mwIndex *ptr = mxGetJc(pm);
  const mwSize *dims = mxGetDimensions(pm);

  int nrows = dims[0];
  int ncols = dims[1];
  int nnz = ptr[ncols];

  size_t diaglength = GetScalarFromCellArray<size_t>(data, 1);

  // convert from mwIndex -> int32_t, double -> real
  std::vector<real> vec_val(val, val + nnz);
  std::vector<int32_t> vec_ptr(ptr, ptr + (ncols + 1));
  std::vector<int32_t> vec_ind(ind, ind + nnz); 

  return BlockIdKronSparse<real>::CreateFromCSC(
    row, 
    col,
    diaglength,
    nrows,
    ncols,
    nnz,
    vec_val,
    vec_ptr,
    vec_ind);
}

prost::BlockIdKronDense<real>*
CreateBlockIdKronDense(size_t row, size_t col, const mxArray *pm)
{
  size_t nrows = mxGetM(mxGetCell(pm, 0));
  size_t ncols = mxGetN(mxGetCell(pm, 0));
  double *data = reinterpret_cast<double *>(mxGetData(mxGetCell(pm, 0)));
  std::vector<real> r_data(data, data + nrows * ncols);
  size_t diaglength = GetScalarFromCellArray<size_t>(pm, 1);

  return BlockIdKronDense<real>::CreateFromColFirstData(diaglength, row, col, nrows, ncols, r_data);
}

prost::BlockDenseKronId<real>*
CreateBlockDenseKronId(size_t row, size_t col, const mxArray *pm)
{
  size_t nrows = mxGetM(mxGetCell(pm, 0));
  size_t ncols = mxGetN(mxGetCell(pm, 0));
  double *data = reinterpret_cast<double *>(mxGetData(mxGetCell(pm, 0)));
  std::vector<real> r_data(data, data + nrows * ncols);
  size_t diaglength = GetScalarFromCellArray<size_t>(pm, 1);

  return BlockDenseKronId<real>::CreateFromColFirstData(diaglength, row, col, nrows, ncols, r_data);
}

prost::BlockSparseKronId<real>*
CreateBlockSparseKronId(size_t row, size_t col, const mxArray *data)
{
  mxArray *pm = mxGetCell(data, 0);

  if(!mxIsSparse(pm))
    throw Exception("Matrix must be sparse!");

  double *val = mxGetPr(pm);
  mwIndex *ind = mxGetIr(pm);
  mwIndex *ptr = mxGetJc(pm);
  const mwSize *dims = mxGetDimensions(pm);

  int nrows = dims[0];
  int ncols = dims[1];
  int nnz = ptr[ncols];

  size_t diaglength = GetScalarFromCellArray<size_t>(data, 1);

  // convert from mwIndex -> int32_t, double -> real
  std::vector<real> vec_val(val, val + nnz);
  std::vector<int32_t> vec_ptr(ptr, ptr + (ncols + 1));
  std::vector<int32_t> vec_ind(ind, ind + nnz); 

  return BlockSparseKronId<real>::CreateFromCSC(
    row, 
    col,
    diaglength,
    nrows,
    ncols,
    nnz,
    vec_val,
    vec_ptr,
    vec_ind);
}
  
BlockZero<real>* 
CreateBlockZero(size_t row, size_t col, const mxArray *data)
{
  size_t nrows = GetScalarFromCellArray<size_t>(data, 0);
  size_t ncols = GetScalarFromCellArray<size_t>(data, 1);
  
  return new BlockZero<real>(row, col, nrows, ncols);
}
  
BackendPDHG<real>* 
CreateBackendPDHG(const mxArray *data)
{
  BackendPDHG<real>::Options opts;

  // read options from data
  opts.tau0 =                 GetScalarFromField<real>(data, "tau0");
  opts.sigma0 =               GetScalarFromField<real>(data, "sigma0");
  opts.residual_iter =        GetScalarFromField<int>(data,  "residual_iter"); 
  opts.scale_steps_operator = GetScalarFromField<bool>(data, "scale_steps_operator");
  opts.alg2_gamma =           GetScalarFromField<real>(data, "alg2_gamma");
  opts.arg_alpha0 =           GetScalarFromField<real>(data, "arg_alpha0");
  opts.arg_nu =               GetScalarFromField<real>(data, "arg_nu");
  opts.arg_delta =            GetScalarFromField<real>(data, "arg_delta");
  opts.arb_delta =            GetScalarFromField<real>(data, "arb_delta");
  opts.arb_tau =              GetScalarFromField<real>(data, "arb_tau");

  std::string stepsize_variant(mxArrayToString(mxGetField(data, 0, "stepsize")));

  if(stepsize_variant == "alg1")
    opts.stepsize_variant = BackendPDHG<real>::StepsizeVariant::kPDHGStepsAlg1;
  else if(stepsize_variant == "alg2")
    opts.stepsize_variant= BackendPDHG<real>::StepsizeVariant::kPDHGStepsAlg2;
  else if(stepsize_variant == "goldstein")
    opts.stepsize_variant= BackendPDHG<real>::StepsizeVariant::kPDHGStepsResidualGoldstein;
  else if(stepsize_variant == "boyd")
    opts.stepsize_variant= BackendPDHG<real>::StepsizeVariant::kPDHGStepsResidualBoyd;
  else
    throw Exception("Couldn't recognize step-size variant. Valid options are {alg1,alg2,goldstein,boyd}.");

  BackendPDHG<real> *backend = new BackendPDHG<real>(opts);

  return backend;
}

BackendADMM<real>* 
CreateBackendADMM(const mxArray *data)
{
  BackendADMM<real>::Options opts;

  // read options from data
  opts.rho0 =           GetScalarFromField<real>(data, "rho0");
  opts.residual_iter =  GetScalarFromField<int>(data,  "residual_iter"); 
  opts.arb_delta =      GetScalarFromField<real>(data, "arb_delta");
  opts.arb_gamma =      GetScalarFromField<real>(data, "arb_gamma");
  opts.arb_tau =        GetScalarFromField<real>(data, "arb_tau");
  opts.alpha =          GetScalarFromField<real>(data, "alpha");
  opts.cg_max_iter =    GetScalarFromField<int>(data, "cg_max_iter");
  opts.cg_tol_pow  =    GetScalarFromField<real>(data, "cg_tol_pow");
  opts.cg_tol_min  =    GetScalarFromField<real>(data, "cg_tol_min");
  opts.cg_tol_max  =    GetScalarFromField<real>(data, "cg_tol_max");

  BackendADMM<real> *backend = new BackendADMM<real>(opts);

  return backend;
}
    
std::shared_ptr<Prox<real> >
CreateProx(const mxArray *pm) 
{
  const mwSize *dims = mxGetDimensions(pm);
  const mwSize max_dim = std::max(dims[0], dims[1]);

  if(max_dim != 5)
  {
    std::stringstream ss;
    ss << "Invalid prox description. Dim = " << max_dim << " (should be 5).";
    throw Exception(ss.str());
  }

  std::string name(mxArrayToString(mxGetCell(pm, 0)));
  size_t idx = GetScalarFromCellArray<size_t>(pm, 1);
  size_t size = GetScalarFromCellArray<size_t>(pm, 2);
  bool diagsteps = GetScalarFromCellArray<bool>(pm, 3);
  mxArray *data = mxGetCell(pm, 4);
  
  Prox<real> *prox = nullptr;
  
  try
  {
    for(auto& p : get_prox_reg())
      if(p.first.compare(name) == 0)
        prox = p.second(idx, size, diagsteps, data);
  }
  catch(Exception& e)
  {
    std::ostringstream ss;
    ss << "Creating prox with ID '" << name << "' failed. Reason: " << e.what();
    throw Exception(ss.str());
  }

  if(!prox) // prox with that name does not exist
  {
    std::ostringstream ss;
    ss << "Creating prox with ID '" << name << "' failed. Reason: Name not registered in ProxFactory.";
    ss << " Available prox are: { ";
    for(auto& b : get_prox_reg())
      ss << b.first << ", ";
    ss.seekp(-2, ss.cur); ss << " }." << std::endl;

    throw Exception(ss.str());
  }

  return std::shared_ptr<Prox<real> >(prox);
}

std::shared_ptr<Block<real> >
CreateBlock(const mxArray *pm)
{
  const mwSize *dims = mxGetDimensions(pm);
  const mwSize max_dim = std::max(dims[0], dims[1]);

  if(max_dim != 4)
    throw Exception("Invalid block description. Dim != 4.");
  
  std::string name(mxArrayToString(mxGetCell(pm, 0)));

  size_t row = GetScalarFromCellArray<size_t>(pm, 1);
  size_t col = GetScalarFromCellArray<size_t>(pm, 2);
  mxArray *data = mxGetCell(pm, 3);

  Block<real> *block = nullptr;

  try
  {
    for(auto& b : get_block_reg())
      if(b.first.compare(name) == 0)
        block = b.second(row, col, data);
  }
  catch(Exception& e)
  {
    std::ostringstream ss;
    ss << "Creating block with ID '" << name << "' failed. Reason: " << e.what();
    throw Exception(ss.str());
  }

  if(!block) // block with that name does not exist
  {
    std::ostringstream ss;
    ss << "Creating block with ID '" << name << "' failed. Reason: Name not registered in BlockFactory.";
    ss << " Available blocks are: { ";
    for(auto& b : get_block_reg())
      ss << b.first << ", ";
    ss.seekp(-2, ss.cur); ss << " }." << std::endl;

    throw Exception(ss.str());
  }

  return std::shared_ptr<Block<real> >(block);
}
    
std::shared_ptr<Backend<real> >
CreateBackend(const mxArray *pm)
{
  std::string name(mxArrayToString(mxGetCell(pm, 0)));
  std::transform(name.begin(), name.end(), name.begin(), ::tolower);

  Backend<real> *backend = nullptr;

  try
  {
    for(auto& b : default_backend_reg)
      if(b.first.compare(name) == 0)
        backend = b.second(mxGetCell(pm, 1));
  }
  catch(Exception& e)
  {
    std::ostringstream ss;
    ss << "Creating backend with ID '" << name << "' failed. Reason: " << e.what();
    throw Exception(ss.str());
  }

  if(!backend)
  {
    std::ostringstream ss;
    ss << "Creating backend with ID '" << name << "' failed. Reason: Name not registered in BackendFactory.";
    ss << " Available backends are: { ";
    for(auto& b : default_backend_reg)
      ss << b.first << ", ";
    ss.seekp(-2, ss.cur); ss << " }." << std::endl;

    throw Exception(ss.str());
  }

  return std::shared_ptr<Backend<real> >(backend);
}

std::shared_ptr<Problem<real> >
CreateProblem(const mxArray *pm, size_t nrows, size_t ncols)
{
  Problem<real> *prob = new Problem<real>;

  std::vector<const mxArray *> blocks = GetCellArray(mxGetField(pm, 0, "linop"));
  std::vector<const mxArray *> prox_g = GetCellArray(mxGetField(pm, 0, "prox_g"));
  std::vector<const mxArray *> prox_f = GetCellArray(mxGetField(pm, 0, "prox_f"));
  std::vector<const mxArray *> prox_gstar = GetCellArray(mxGetField(pm, 0, "prox_gstar"));
  std::vector<const mxArray *> prox_fstar = GetCellArray(mxGetField(pm, 0, "prox_fstar"));

  // add blocks
  for(auto& b : blocks) prob->AddBlock(CreateBlock(b));

  // add proxs
  for(auto& p : prox_g) prob->AddProx_g(CreateProx(p));
  for(auto& p : prox_f) prob->AddProx_f(CreateProx(p));
  for(auto& p : prox_gstar) prob->AddProx_gstar(CreateProx(p));
  for(auto& p : prox_fstar) prob->AddProx_fstar(CreateProx(p));

  // set scaling
  std::string scaling(mxArrayToString(mxGetField(pm, 0, "scaling")));

  if(scaling == "alpha")
    prob->SetScalingAlpha( GetScalarFromField<real>(pm, "scaling_alpha") );
  else if(scaling == "identity")
    prob->SetScalingIdentity();
  else if(scaling == "custom")
  {
    std::vector<real> left = GetVector<real>(mxGetField(pm, 0, "scaling_left"));
    std::vector<real> right = GetVector<real>(mxGetField(pm, 0, "scaling_right"));

    prob->SetScalingCustom(left, right);
  }
  else
    throw Exception("Problem scaling variant not recognized. Options are {'alpha', 'identity', 'custom'}.");

  prob->SetDimensions(nrows, ncols);

  return std::shared_ptr<Problem<real> >(prob);
}

Solver<real>::Options 
CreateSolverOptions(const mxArray *pm)
{
  Solver<real>::Options opts;

  opts.tol_rel_primal =     GetScalarFromField<real>(pm, "tol_rel_primal");
  opts.tol_rel_dual =       GetScalarFromField<real>(pm, "tol_rel_dual");
  opts.tol_abs_primal =     GetScalarFromField<real>(pm, "tol_abs_primal");
  opts.tol_abs_dual =       GetScalarFromField<real>(pm, "tol_abs_dual");
  opts.max_iters =          GetScalarFromField<int>(pm,  "max_iters");
  opts.num_cback_calls =    GetScalarFromField<int>(pm,  "num_cback_calls");
  opts.verbose =            GetScalarFromField<bool>(pm, "verbose");
  opts.solve_dual_problem = GetScalarFromField<bool>(pm, "solve_dual");

  if(mxGetM(mxGetField(pm, 0, "x0")) > 0) opts.x0 = GetVector<real>(mxGetField(pm, 0, "x0"));
  if(mxGetM(mxGetField(pm, 0, "y0")) > 0) opts.y0 = GetVector<real>(mxGetField(pm, 0, "y0"));

  Solver_interm_cb_handle = mxGetField(pm, 0, "interm_cb");

  return opts;
}

map<string, function<prost::Prox<real>*(size_t, size_t, bool, const mxArray*)>>& get_prox_reg()
{
  static map<string, function<Prox<real>*(size_t, size_t, bool, const mxArray*)>> prox_reg;

  return prox_reg;
}
  
map<string, function<prost::Block<real>*(size_t, size_t, const mxArray*)>>& get_block_reg()
{
  static map<string, function<Block<real>*(size_t, size_t, const mxArray*)>> block_reg;

  return block_reg;
}

struct InitRegistries {
  InitRegistries() {
    get_prox_reg().insert(default_prox_reg.begin(), default_prox_reg.end());
    get_block_reg().insert(default_block_reg.begin(), default_block_reg.end());
  }
};

static InitRegistries initRegistries;

} // namespace matlab
