#include "prost/prox/prox_elem_operation.hpp"
#include "prost/prox/prox_elem_operation.inl"

#include "prost/prox/elemop/elem_operation.hpp"
#include "prost/prox/elemop/elem_operation_1d.hpp"
#include "prost/prox/elemop/elem_operation_norm2.hpp"
#include "prost/prox/elemop/elem_operation_ind_simplex.hpp"
#include "prost/prox/elemop/elem_operation_singular_nx2.hpp"
#include "prost/prox/elemop/function_1d.hpp"
#include "prost/prox/elemop/function_2d.hpp"

namespace prost {
  
// Explicit template instantiation

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

} // namespace prost
