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

#include "prost/prox/prox_elem_operation.hpp"
#include "prost/prox/prox_elem_operation.inl"

#include "prost/prox/elemop/elem_operation.hpp"
#include "prost/prox/elemop/elem_operation_1d.hpp"
#include "prost/prox/elemop/elem_operation_norm2.hpp"
#include "prost/prox/elemop/elem_operation_ind_simplex.hpp"
#include "prost/prox/elemop/elem_operation_ind_sum.hpp"
#include "prost/prox/elemop/elem_operation_singular_nx2.hpp"
#include "prost/prox/elemop/elem_operation_ind_psd_cone_3x3.hpp"
#include "prost/prox/elemop/elem_operation_mass_norm.hpp"
#include "prost/prox/elemop/function_1d.hpp"
#include "prost/prox/elemop/function_2d.hpp"

namespace prost {
  
// Explicit template instantiation
// TODO: simplify this

// float
// ElemOperation1D
template class ProxElemOperation<float, ElemOperation1D<float, Function1DZero<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DAbs<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DSquare<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndLeq0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndGeq0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndEq0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DIndBox01<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DMaxPos0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DL0<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DHuber<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DLq<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DLqPlusEps<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DTruncQuad<float>>>;
template class ProxElemOperation<float, ElemOperation1D<float, Function1DTruncLinear<float>>>;

// ElemOperationNorm2
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DZero<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DAbs<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DSquare<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndLeq0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndGeq0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndEq0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DIndBox01<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DMaxPos0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DL0<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DHuber<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DLq<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DLqPlusEps<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DTruncQuad<float>>>;
template class ProxElemOperation<float, ElemOperationNorm2<float, Function1DTruncLinear<float>>>;


// ElemOperationSingularNx2<Function2DSum1D>
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DZero<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DAbs<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DSquare<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DIndLeq0<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DIndGeq0<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DIndEq0<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DIndBox01<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DMaxPos0<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DL0<float>>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DSum1D<float, Function1DHuber<float>>>>;

// ElemOperationSingularNx2<other>
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DIndL1Ball<float>>>;
template class ProxElemOperation<float, ElemOperationSingularNx2<float, Function2DMoreau<float, Function2DIndL1Ball<float>>>>;

// Other
template class ProxElemOperation<float, ElemOperationIndSimplex<float>>;
template class ProxElemOperation<float, ElemOperationIndSum<float>>;
template class ProxElemOperation<float, ElemOperationIndPsdCone3x3<float>>;
template class ProxElemOperation<float, ElemOperationMass4<float,false>>;
template class ProxElemOperation<float, ElemOperationMass5<float,false>>;
template class ProxElemOperation<float, ElemOperationMass4<float,true>>;
template class ProxElemOperation<float, ElemOperationMass5<float,true>>;

// double
// ElemOperation1D 
template class ProxElemOperation<double, ElemOperation1D<double, Function1DZero<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DAbs<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DSquare<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndLeq0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndGeq0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndEq0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DIndBox01<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DMaxPos0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DL0<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DHuber<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DLq<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DLqPlusEps<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DTruncQuad<double>>>;
template class ProxElemOperation<double, ElemOperation1D<double, Function1DTruncLinear<double>>>;

// ElemOperationNorm2
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DZero<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DAbs<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DSquare<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndLeq0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndGeq0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndEq0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DIndBox01<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DMaxPos0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DL0<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DHuber<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DLq<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DLqPlusEps<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DTruncQuad<double>>>;
template class ProxElemOperation<double, ElemOperationNorm2<double, Function1DTruncLinear<double>>>;

// ElemOperationSingularNx2<Function2DSum1D>
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DZero<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DAbs<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DSquare<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DIndLeq0<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DIndGeq0<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DIndEq0<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DIndBox01<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DMaxPos0<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DL0<double>>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DSum1D<double, Function1DHuber<double>>>>;

// ElemOperationSingularNx2<other>
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DIndL1Ball<double>>>;
template class ProxElemOperation<double, ElemOperationSingularNx2<double, Function2DMoreau<double, Function2DIndL1Ball<double>>>>;


// other
template class ProxElemOperation<double, ElemOperationIndSimplex<double>>;
template class ProxElemOperation<double, ElemOperationIndSum<double>>;
template class ProxElemOperation<double, ElemOperationIndPsdCone3x3<double>>;
template class ProxElemOperation<double, ElemOperationMass4<double, true>>;
template class ProxElemOperation<double, ElemOperationMass4<double, false>>;
template class ProxElemOperation<double, ElemOperationMass5<double, true>>;
template class ProxElemOperation<double, ElemOperationMass5<double, false>>;

} // namespace prost
