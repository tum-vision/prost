#ifndef LINOP_HPP_
#define LINOP_HPP_

/*
 * @brief Abstract base class for linear operator blocks. 
 *
 */
template<typename T>
class LinOp {
 public:
  LinOp(size_t row, size_t col, size_t nrows, size_t ncols);
  virtual ~LinOp();

  virtual bool Init();
  virtual void Release();
  
  void EvalAdd(T *d_res, T *d_rhs);
  void EvalAdjointAdd(T *d_res, T *d_rhs);
  
 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);
  
  size_t row_;
  size_t col_;
  size_t nrows_;
  size_t ncols_;
};

/*
 * @brief Block matrix built out of linops.
 *
 */

template<typename T>
class LinearOperator {
 public:
  
  
 protected:
  std::vector<std::vector<LinOp*> > block_rows_;
};

#endif
