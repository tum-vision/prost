#ifndef LINOP_DATA_GRAPH_PREC_
#define LINOP_DATA_GRAPH_PREC_

#include "linop.hpp"

/**
 * @brief Assumes pixel-first (matlab style, rows first) ordering and
 *        outputs dx dy pixelwise ordered.
 *
 */
template<typename T>
class LinOpDataGraphPrec : public LinOp<T> {
 public:
  LinOpDataGraphPrec(size_t row, size_t col, size_t nx, size_t ny, size_t L, T left, T right);
  virtual ~LinOpDataGraphPrec();

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const;
  
  virtual T col_sum(size_t col, T alpha) const;
  
 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels
  
  T t_min_;
  T t_max_;
};

#endif
