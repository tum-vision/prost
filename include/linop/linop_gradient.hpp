#ifndef LINOP_GRADIENT_HPP_
#define LINOP_GRADIENT_HPP_

#include "linop.hpp"

/**
 * @brief Assumes label-first ordering and outputs dx dy pixelwise ordered.
 *
 */
template<typename T>
class LinOpGradient2D : public LinOp<T> {
 public:
  LinOpGradient2D(size_t row, size_t col, size_t nx, size_t ny, size_t L);
  virtual ~LinOpGradient2D();

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const { return 2; }
  virtual T col_sum(size_t col, T alpha) const { return 4; }
  
 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels
};

/**
 * @brief Assumes label-first ordering and outputs dx dy dt pixelwise ordered.
 *
 */
template<typename T>
class LinOpGradient3D : public LinOp<T> {
 public:
  LinOpGradient3D(size_t row, size_t col, size_t nx, size_t ny, size_t L);
  virtual ~LinOpGradient3D();

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const { return 2; }
  virtual T col_sum(size_t col, T alpha) const { return 6; }

 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels
};

#endif
