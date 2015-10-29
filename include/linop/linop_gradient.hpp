#ifndef LINOP_GRADIENT_HPP_
#define LINOP_GRADIENT_HPP_

#include "linop.hpp"

/**
 * @brief Assumes pixel-first (matlab style, rows first) ordering and
 *        outputs dx dy pixelwise ordered.
 * 
 *        If label_first == true, assumes label first then rows then cols 
 *        ordering but still outputs dx dy pixelwise ordered.
 */
template<typename T>
class LinOpGradient2D : public LinOp<T> {
 public:
  LinOpGradient2D(
    size_t row, 
    size_t col, 
    size_t nx, 
    size_t ny, 
    size_t L, 
    bool label_first = false,
    const std::vector<T>& m11 = std::vector<T>(), 
    const std::vector<T>& m12 = std::vector<T>(), 
    const std::vector<T>& m22 = std::vector<T>());
  virtual ~LinOpGradient2D();

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const; // { return 2; }
  virtual T col_sum(size_t col, T alpha) const; // { return 4; }
  
 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels
  bool label_first_; // ordering of the image

  std::vector<T> m11_, m12_, m22_;
  T *d_m11_, *d_m12_, *d_m22_; // weighted gradient
};

/**
 * @brief Assumes pixel-first (matlab style, rows first) ordering and
 *        outputs dx dy dt pixelwise ordered.
 *
 */
template<typename T>
class LinOpGradient3D : public LinOp<T> {
 public:
  LinOpGradient3D(
    size_t row, 
    size_t col, 
    size_t nx, 
    size_t ny, 
    size_t L,
    const std::vector<T>& m11 = std::vector<T>(), 
    const std::vector<T>& m12 = std::vector<T>(), 
    const std::vector<T>& m22 = std::vector<T>());

  virtual ~LinOpGradient3D();

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const; // { return 2; }
  virtual T col_sum(size_t col, T alpha) const; // { return 6; }

 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels

  std::vector<T> m11_, m12_, m22_;
  T *d_m11_, *d_m12_, *d_m22_; // weighted gradient
};

#endif
