#ifndef LINOP_HPP_
#define LINOP_HPP_

#include <cstdlib>
#include <vector>

/*
 * @brief Abstract base class for linear operator blocks. Also
 *        implements the "zero" operator.
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

  // required for preconditioners
  // row and col are "local" for the operator, which means they start at 0
  virtual T row_sum(size_t row, T alpha) const { return 0; }
  virtual T col_sum(size_t col, T alpha) const { return 0; }

  size_t row() const { return row_; }
  size_t col() const { return col_; }
  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; }

  virtual size_t gpu_mem_amount() const { return 0; }
  
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
  LinearOperator();
  virtual ~LinearOperator();

  // careful: transfers ownership to LinearOperator
  void AddOperator(LinOp<T> *op);
  
  bool Init();
  void Release();

  void Eval(T *d_res, T *d_rhs);
  void EvalAdjoint(T *d_res, T *d_rhs);

  // required for preconditioners
  T row_sum(size_t row, T alpha) const;
  T col_sum(size_t col, T alpha) const;

  size_t nrows() const { return nrows_; }
  size_t ncols() const { return ncols_; } 

  size_t gpu_mem_amount() const;
  
 protected:
  std::vector<LinOp<T>*> operators_;
  size_t nrows_;
  size_t ncols_;
};

#endif
