/*
 * This file is part of pdsolver.
 *
 * Copyright (C) 2015 Thomas Möllenhoff <thomas.moellenhoff@in.tum.de> 
 *
 * pdsolver is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * pdsolver is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with pdsolver. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "prox_create.h"

#include "../pdsolver/prox/prox.h"
#include "../pdsolver/prox/prox_sum_1d.h"
#include "../pdsolver/prox/prox_sum_norm2.h"
#include "../pdsolver/prox/prox_moreau.h"
#include "../pdsolver/prox/prox_fun_1d.h"

#include <string>

void read_prox_coeffs(prox_1d_coefficients &coeffs, const mxArray *ptr) {

  const mwSize *dims;
  double *val;

  std::vector<real>* coeff_array[kNumCoeffs] = {
    &coeffs.a,
    &coeffs.b,
    &coeffs.c,
    &coeffs.d,
    &coeffs.e };

  // starts at 1 because cell 0 is prox-name.
  for(int i = 1; i <= kNumCoeffs; ++i)
  {
    dims = mxGetDimensions(mxGetCell(ptr, i));
    val = mxGetPr(mxGetCell(ptr, i));

    for(int j = 0; j < dims[0]; j++)
      (*coeff_array[i - 1]).push_back( (real) val[j] );
  }
}

prox_sum_1d *create_prox_sum_1d(int idx,
                                int count,
                                const mxArray *data)
{
  std::string name(mxArrayToString(mxGetCell(data, 0)));
  prox_fun_1d fn = prox_fun_from_string(name);

  if(kUnknownProx == fn)
    return 0;

  prox_1d_coefficients prox_coeffs;
  read_prox_coeffs(prox_coeffs, data);

  return new prox_sum_1d(idx, count, prox_coeffs, fn);
}

prox_sum_norm2 *create_prox_sum_norm2(int idx,
                                      int count,
                                      int dim,
                                      bool interleaved,
                                      const mxArray *data)
{
  std::string name(mxArrayToString(mxGetCell(data, 0)));
  prox_fun_1d fn = prox_fun_from_string(name);

  if(kUnknownProx == fn)
    return 0;

  prox_1d_coefficients prox_coeffs;
  read_prox_coeffs(prox_coeffs, data);
  
  return new prox_sum_norm2(idx, count, dim, interleaved, prox_coeffs, fn);
}

bool prox_from_matlab(const mxArray *pm, std::vector<prox *>& result) {
  const mwSize *dims = mxGetDimensions(pm);
  int num_proxes = dims[0];

  for(int i = 0; i < num_proxes; i++) {
    mxArray *prox_cell = mxGetCell(pm, i);
    
    std::string name(mxArrayToString(mxGetCell(prox_cell, 0)));
    transform(name.begin(), name.end(), name.begin(), ::tolower);

    int idx = (int) mxGetScalar(mxGetCell(prox_cell, 1));
    int count = (int) mxGetScalar(mxGetCell(prox_cell, 2));
    int dim = (int) mxGetScalar(mxGetCell(prox_cell, 3));
    bool interleaved = (bool) mxGetScalar(mxGetCell(prox_cell, 4));
    bool diagprox = (bool) mxGetScalar(mxGetCell(prox_cell, 5));
    mxArray *data = mxGetCell(prox_cell, 6);

    mexPrintf("Attempting to create prox<'%s',idx=%d,cnt=%d,dim=%d,inter=%d,diag=%d>...",
              name.c_str(), idx, count, dim, interleaved, diagprox);
    
    prox *p = 0;
    if("sum_1d" == name)
      p = create_prox_sum_1d(idx, count, data);
    else if("sum_norm2" == name)
      p = create_prox_sum_norm2(idx, count, dim, interleaved, data);

    if(0 == p) // prox not recognized.
    {
      mexPrintf(" invalid argument!\n");
      result.clear();
      
      return false;
    }

    result.push_back(p);
    mexPrintf(" done!\n");
  }

  return true;
}
