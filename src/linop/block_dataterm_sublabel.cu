#include "prost/linop/block_dataterm_sublabel.hpp"

namespace prost {

template<typename T>
BlockDatatermSublabel<T>::BlockDatatermSublabel(
  size_t row, 
  size_t col, 
  size_t nx, 
  size_t ny, 
  size_t L, 
  T left, 
  T right)
  : Block<T>(0,0,0,0)
{
}

template<typename T>
T BlockDatatermSublabel<T>::row_sum(size_t row, T alpha) const
{
  return 0;
}

template<typename T>
T BlockDatatermSublabel<T>::col_sum(size_t col, T alpha) const
{
  return 0;
}

template<typename T>
void BlockDatatermSublabel<T>::EvalLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end)
{
}

template<typename T>
void BlockDatatermSublabel<T>::EvalAdjointLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end)
{
}

// Explicit template instantiation
template class BlockDatatermSublabel<float>;
template class BlockDatatermSublabel<double>;
  
} // namespace prost 