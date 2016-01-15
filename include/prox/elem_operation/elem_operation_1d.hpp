#ifndef ELEM_OPERATION_1D_HPP_
#define ELEM_OPERATION_1D_HPP_


/**
 * @brief Provides proximal operator for fully separable 1D functions:
 * 
 *        sum_i c_i * f(a_i x - b_i) + d_i x + (e_i / 2) x^2.
 *
 *        alpha and beta are generic parameters depending on the choice of f,
 *        e.g. the power for f(x) = |x|^{alpha}.
 *
 */
namespace prox {
namespace elemOperation {
template<typename T, class OPERATION_1D>
struct ElemOperation1D : public ElemOperation<1> {
 struct Data {
    T a, b, c, d, e, alpha, beta;
 };

 typedef Vec ProxSeparableSum<T, ElemOperation1D>::Vector;
 
 
 inline __device__ void operator()(Vec& arg, Vec& res, Data& data) { res[0] = do something with OPERATION_1D.Eval(arg[0], data.alpha, data.beta); }
};
}
}

#endif
