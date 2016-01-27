#include "linop/block.hpp"

using namespace linop;

template<typename T>
Block<T>::Block(size_t row, size_t col, size_t nrows, size_t ncols)
    : row_(row), col_(col), nrows_(nrows), ncols_(ncols) {}

template<typename T>
Block<T>::~Block() {
    Release();
}
    
template<typename T>
void Block<T>::EvalAdd(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs) {
  EvalLocalAdd(res.begin() + row_,
               res.begin() + row_ + nrows_,
               rhs.begin() + col_,
               rhs.begin() + col_ + ncols_);
}

template<typename T>
void Block<T>::EvalAdjointAdd(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs) {
  EvalAdjointLocalAdd(res.begin() + col_,
                      res.begin() + col_ + ncols_,
                      rhs.begin() + row_,
                      rhs.begin() + row_ + nrows_);
}

template<typename T>
void Block<T>::Init() {}

template<typename T>
void Block<T>::Release() {}


// Explicit template instantiation
template class Block<float>;
template class Block<double>;

