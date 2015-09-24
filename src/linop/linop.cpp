#include "linop.hpp"

template<typename T>
LinOp<T>::LinOp(size_t row, size_t col, size_t nrows, size_t ncols)
    : row_(row), col_(col), nrows_(nrows), ncols_(ncols)
{
}

template<typename T>
LinOp<T>::~LinOp() {
}

template<typename T>
void LinOp<T>::Eval(T *d_res, T *d_rhs) {
  EvalLocal(&d_res[row_], &d_rhs[row_]);
}

template<typename T>
void LinOp<T>::EvalAdjoint(T *d_res, T *d_rhs) {
  EvalAdjointLocal(&d_res[], &d_rhs[]);
}

template<typename T>
void LinOp<T>::EvalLocal(T *d_res, T *d_rhs) {
}

template<typename T>
void LinOp<T>::EvalAdjointLocal(T *d_res, T *d_rhs) {
}

template<typename T>
bool LinOp<T>::Init() {
  return true;
}

template<typename T>
void LinOp<T>::Release() {
}

// Explicit template instantiation
template class LinOp<float>;
template class LinOp<double>;
