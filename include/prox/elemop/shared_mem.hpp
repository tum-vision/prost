#ifndef SHARED_MEM_HPP_
#define SHARED_MEM_HPP_

template<class ELEM_OPERATION>
class SharedMem {
public:
  __device__ SharedMem(size_t threadIdx_x) : threadIdx_x_(threadIdx_x) {
      extern __shared__ char sh_mem[];
      sh_arg_ = reinterpret_cast<typename ELEM_OPERATION::shared_mem_type*>(sh_mem);
  }

  inline __device__ typename ELEM_OPERATION::shared_mem_type operator[](size_t i) const {
    // Out of bounds check?
    size_t index = threadIdx_x_ * ELEM_OPERATION::shared_mem_count + i;
    return sh_arg_[index];
  }

  inline __device__ typename ELEM_OPERATION::shared_mem_type& operator[](size_t i) {
    // Out of bounds check?
    size_t index = threadIdx_x_ * ELEM_OPERATION::shared_mem_count + i;
    return sh_arg_[index];
  }

private:
  size_t threadIdx_x_;
  typename ELEM_OPERATION::shared_mem_type* sh_arg_;
};
#endif


