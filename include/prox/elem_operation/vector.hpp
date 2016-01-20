#ifndef VECTOR_HPP_
#define VECTOR_HPP_

template<typename T, class ELEM_OPERATION>
class Vector {
public:
  Vector(size_t count, bool interleaved, size_t tx, T* data) : 
    count_(count),
    interleaved_(interleaved),
    tx_(tx),
    data_(data) {}

  inline __device__ T operator[](size_t i) const {
    // Out of bounds check?
    size_t index = parent_.interleaved_ ? (tx * ELEM_OPERATION::dim + i) : (tx + count_ * i);
    return data[index];
  }

  inline __device__ T& operator[](size_t i) {
    // Out of bounds check?
    size_t index = parent_.interleaved_ ? (tx * ELEM_OPERATION::dim + i) : (tx + count_ * i);
    return data[index];
  }

private:
  size_t count_;
  bool interleaved_;
  size_t tx_;
  T* data_;
};

#endif


