/**
* This file is part of prost.
*
* Copyright 2016 Thomas Möllenhoff <thomas dot moellenhoff at in dot tum dot de> 
* and Emanuel Laude <emanuel dot laude at in dot tum dot de> (Technical University of Munich)
*
* prost is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* prost is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with prost. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PROST_COMMON_HPP_
#define PROST_COMMON_HPP_

#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <vector>
#include <string>
#include <sstream>

namespace prost {

// common functions from standard library
using std::cout;
using std::endl;
using std::function;
using std::list;
using std::map;
using std::shared_ptr;
using std::string;
using std::stringstream;
using std::vector;

string get_version(); 
template<typename T> list<double> linspace(T start_in, T end_in, int num_in);
int _ConvertSMVer2Cores(int major, int minor);

/// \brief Helper function that converts CSR format to CSC format, 
///        not in-place, if a == NULL, only pattern is reorganized
///        the size of matrix is n x m.
template<typename T>
void csr2csc(int n, int m, int nz, 
             T *a, int *col_idx, int *row_start,
             T *csc_a, int *row_idx, int *col_start); 

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif

} // namespace prost

#endif // PROST_COMMON_HPP_
