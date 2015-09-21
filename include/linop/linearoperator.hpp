#ifndef LINEAROPERATOR_HPP_
#define LINEAROPERATOR_HPP_

/*
 * @brief ...
 *
 */
template<typename T>
class LinearOperator {
public:
  LinearOperator();
  virtual ~LinearOperator();

  virtual void Apply(T *d_res, T *d_rhs) = 0;
  virtual void ApplyAdjoint(T *d_res, T *d_rhs) = 0;
  
protected:
  int index_row, index_col;
  int nrows, ncols;
};

#endif
