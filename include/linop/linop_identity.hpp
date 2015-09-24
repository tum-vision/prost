#ifndef LINOP_IDENTITY_HPP_
#define LINOP_IDENTITY_HPP_

/**
 * @brief Linear operator implementation of the MATLAB command speye.
 *
 * @param ndiags: number of diagonals
 * @param offsets: array of size ndiags, starting position of diagonals
 * @param factors: array of size ndiags, constant factor each diagonal
 *                 is multiplied with
 */
template<typename T>
class LinOpIdentity : public LinOp<T> {
 public:
  LinOpIdentity(size_t row,
                size_t col,
                size_t nrows,
                size_t ncols,
                size_t ndiags,
                const std::vector<size_t>& offsets,
                const std::vector<T>& factors);
  
  virtual ~LinOpIdentity();

  virtual bool Init();
  virtual void Release();

 protected:
  virtual void EvalLocalAdd(T *d_res, T *d_rhs);
  virtual void EvalAdjointLocalAdd(T *d_res, T *d_rhs);

  size_t cmem_offset_;
  size_t ndiags_;
  std::vector<size_t> offsets_;
  std::vector<float> factors_;

  static size_t cmem_counter_;
};

#endif
