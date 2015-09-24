#ifndef LINOP_GRADIENT_HPP_
#define LINOP_GRADIENT_HPP_

/**
 * @brief Assumes label-first ordering and outputs dx dy pixelwise ordered.
 *
 */
template<typename T>
class LinOpGradient {
 public:
  LinOpGradient(size_t row, size_t col, size_t nx, size_t ny, size_t L);
  virtual ~LinOpGradient();

 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t nx; // width of image
  size_t ny; // height of image
  size_t L; // number of labels/channels
};

#endif
