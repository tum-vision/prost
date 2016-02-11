#ifndef PROST_SHARED_MEM_HPP_
#define PROST_SHARED_MEM_HPP_

namespace prost {

// TODO: comment me
template<typename T, class F>
class SharedMem 
{
public:
  __device__
  SharedMem(size_t dim, size_t threadIdx_x)
      : dim_(dim), threadIdx_x_(threadIdx_x)
  {
    extern __shared__ char sh_mem[];
    sh_arg_ = reinterpret_cast<T*>(sh_mem);
  }

  inline __device__
  T operator[](size_t i) const
  {
    size_t index = threadIdx_x_ * get_count_fun_(dim_) + i;
    return sh_arg_[index];
  }

  inline __device__
  T& operator[](size_t i)
  {
    // Out of bounds check?
    size_t index = threadIdx_x_ * get_count_fun_(dim_) + i;
    return sh_arg_[index];
  }

private:
  size_t dim_;
  size_t threadIdx_x_;
  T* sh_arg_;
  F get_count_fun_;
};

} // namespace prost

#endif // PROST_SHARED_MEM_HPP_
