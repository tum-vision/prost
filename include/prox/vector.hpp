#ifndef VECTOR_HPP_
#define VECTOR_HPP_

template<typename T>
class Vector
{
public:
  __host__ __device__
  Vector(size_t count, size_t dim, bool interleaved, size_t tx, const T* data) :
    count_(count),
    dim_(dim),
    interleaved_(interleaved),
    tx_(tx),
    data_(nullptr),
    const_data_(data) { }

  __host__ __device__
  Vector(size_t count, size_t dim, bool interleaved, size_t tx, T* data) :
    count_(count),
    dim_(dim),
    interleaved_(interleaved),
    tx_(tx),
    data_(data), 
    const_data_(data) { }

  inline __host__ __device__
  T
  operator[](size_t i) const 
  {
    size_t index = interleaved_ ? (tx_ * dim_ + i) : (tx_ + count_ * i);
    return const_data_[index];
  }

  inline __host__ __device__
  T&
  operator[](size_t i) 
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
  const T* const_data_;
};

#endif
