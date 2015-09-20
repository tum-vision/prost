#ifndef LINEAROPERATOR_HPP_
#define LINEAROPERATOR_HPP_

/*
 * @brief ...
 *
 */
class LinearOperator {
public:
  LinearOperator();
  virtual ~LinearOperator();

  virtual void Apply(real *d_result, real *d_rhs) = 0;
  virtual void ApplyAdjoint(real *d_result, real *d_rhs) = 0;
  
protected:
  int index_row, index_col;
  int nrows, ncols;
};

#endif
