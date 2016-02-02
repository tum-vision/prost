#ifndef VECTOR_HPP_
#define VECTOR_HPP_


namespace prox 
{

template<typename T>
class Vector 
{
public:
  __device__ Vector(size_t count, size_t dim, bool interleaved, size_t tx, T* data) : 
    count_(count),
    dim_(dim),
    interleaved_(interleaved),
    tx_(tx),
    data_(data) {}

  inline __device__ T operator[](size_t i) const 
  {
    // Out of bounds check?
    size_t index = interleaved_ ? (tx_ * dim_ + i) : (tx_ + count_ * i);
    return data_[index];
  }

  inline __device__ T& operator[](size_t i) 
  {
    // Out of bounds check?
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

}

#endif


