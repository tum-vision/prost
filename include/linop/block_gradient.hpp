/**
#ifndef BLOCK_GRADIENT_HPP_
#define BLOCK_GRADIENT_HPP_

#include "block.hpp"

/**
 * @brief Assumes pixel-first (matlab style, rows first) ordering and
 *        outputs dx dy pixelwise ordered.
 * 
 *        If label_first == true, assumes label first then rows then cols 
 *        ordering but still outputs dx dy pixelwise ordered.
 */
namespace linop {

template<typename T>
class BlockGradient2D : public Block<T> {
 public:
  BlockGradient2D(
    size_t row, 
    size_t col, 
    size_t nx, 
    size_t ny, 
    size_t L, 
    bool label_first = false,
    const std::vector<T>& m11 = std::vector<T>(), 
    const std::vector<T>& m12 = std::vector<T>(), 
    const std::vector<T>& m22 = std::vector<T>(),
    T hx = static_cast<T>(1), // grid discretization
    T hy = static_cast<T>(1));
  virtual ~BlockGradient2D();

  virtual bool Init();
  virtual void Release();
  virtual size_t gpu_mem_amount() const;

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
  T *d_m11_, *d_m12_, *d_m22_; // weighted 2d gradient 
  T hx_, hy_; // grid spacing
};

/**
 * @brief Assumes pixel-first (matlab style, rows first) ordering and
 *        outputs dx dy dt pixelwise ordered.
 *
 */
template<typename T>
class BlockGradient3D : public Block<T> {
 public:
  BlockGradient3D(
    size_t row, 
    size_t col, 
    size_t nx, 
    size_t ny, 
    size_t L,
    bool label_first = false,
    const std::vector<T>& m11 = std::vector<T>(), 
    const std::vector<T>& m12 = std::vector<T>(), 
    const std::vector<T>& m22 = std::vector<T>(),
    T hx = static_cast<T>(1),
    T hy = static_cast<T>(1),
    T ht = static_cast<T>(1));

  virtual ~BlockGradient3D();

  virtual bool Init();
  virtual void Release();
  virtual size_t gpu_mem_amount() const;

  // required for preconditioners
  virtual T row_sum(size_t row, T alpha) const; // { return 2; }
  virtual T col_sum(size_t col, T alpha) const; // { return 6; }

 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx_; // width of image
  size_t ny_; // height of image
  size_t L_; // number of labels/channels
  bool label_first_; // ordering of the image

  std::vector<T> m11_, m12_, m22_;
  T *d_m11_, *d_m12_, *d_m22_; // weighted gradient
  T hx_, hy_, ht_;
};
}
#endif
*/