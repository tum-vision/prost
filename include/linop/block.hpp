#ifndef BLOCK_HPP_
#define BLOCK_HPP_

#include <cstdlib>
#include <vector>
#include <thrust/device_vector.h>


/*
 * @brief Abstract base class for linear operator blocks.
 *
 */
namespace linop {
    
template<typename T>
class Block {
public:
  Block(size_t row, size_t col, size_t nrows, size_t ncols);
  
  virtual ~Block();

  virtual void Init();
  virtual void Release();
  
  void EvalAdd(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs);
  void EvalAdjointAdd(thrust::device_vector<T>& res, thrust::device_vector<T>& rhs);

  // required for preconditioners
  // row and col are "local" for the operator, which means they start at 0
  virtual T row_sum(size_t row, T alpha) = 0;
  virtual T col_sum(size_t col, T alpha) = 0;

  size_t row() const { return row_; }
  size_t col() const { return col_; }
  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

  virtual size_t gpu_mem_amount() = 0;
protected:
  virtual void EvalLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end) = 0;
  
  virtual void EvalAdjointLocalAdd(const typename thrust::device_vector<T>::iterator& res_begin,
                            const typename thrust::device_vector<T>::iterator& res_end,
                            const typename thrust::device_vector<T>::iterator& rhs_begin,
                            const typename thrust::device_vector<T>::iterator& rhs_end) = 0;
private:
  size_t row_;
  size_t col_;
  size_t nrows_;
  size_t ncols_;
};
}
#endif