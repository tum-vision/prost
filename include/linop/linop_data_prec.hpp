#ifndef LINOP_DATA_PREC_
#define LINOP_DATA_PREC_

#include "linop.hpp"

/**
 * @brief Assumes pixel-first (matlab style, rows first) ordering and
 *        outputs dx dy pixelwise ordered.
 *
 */
template<typename T>
class LinOpDataPrec : public LinOp<T> {
 public:
  LinOpDataPrec(size_t row, size_t col, size_t nx, size_t ny, size_t L, T left, T right);
  virtual ~LinOpDataPrec();

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const;
  
  virtual T col_sum(size_t col, T alpha) const;
  
 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels
  
  T left_;
  T right_;
};

#endif
