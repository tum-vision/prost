#ifndef PROST_VECTOR_HPP_
#define PROST_VECTOR_HPP_

namespace prost {
///
/// \brief Helper class for proximal operators which abstracts away the
///        different access patterns to global memory / the variables.
///      
template<typename T>
//TODO rename deviceVectorView
class Vector
{
public:
  __host__ __device__
  Vector(size_t count, size_t dim, bool interleaved, size_t tx, T* data) :
    count_(count),
    dim_(dim),
    interleaved_(interleaved),
    tx_(tx),
    data_(data) { }

  inline __host__ __device__
  T&
  operator[](size_t i) const
  {
    size_t index = interleaved_ ? (tx_ * dim_ + i) : (tx_ + count_ * i);
    return data_[index];
  }
  
private:
  size_t count_;
  size_t dim_;
  bool interleaved_;
  size_t tx_;
  T* data_;
};

} // namespace prost

#endif // PROST_VECTOR_HPP_
