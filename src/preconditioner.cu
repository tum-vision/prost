#include "preconditioner.hpp"

Preconditioner::Preconditioner(SparseMatrix<real> *mat)
    : mat_(mat), d_left_(NULL), d_right_(NULL)
{
}

Preconditioner::~Preconditioner() {
  cudaFree(d_left_);
  cudaFree(d_right_);
}

void Preconditioner::ComputeScalar() {
  type_ = kPrecondScalar;

  real norm = MatrixNormest(*mat_);
  int m = mat_->nrows();
  int n = mat_->ncols();
  
  real *h_left = new real[m];
  real *h_right = new real[n];
  for(int i = 0; i < m; i++) h_left[i] = 1.0 / norm;
  for(int i = 0; i < n; i++) h_right[i] = 1.0 / norm;

  // copy preconditioners to GPU
  cudaMalloc((void **)&d_left_, sizeof(real) * m);
  cudaMalloc((void **)&d_right_, sizeof(real) * n);
  cudaMemcpy(d_left_, h_left, sizeof(real) * m, cudaMemcpyHostToDevice);
  cudaMemcpy(d_right_, h_right, sizeof(real) * n, cudaMemcpyHostToDevice);

  delete [] h_left;
  delete [] h_right;
}

void Preconditioner::ComputeAlpha(real alpha) {
  type_ = kPrecondAlpha;

  // TODO: implement me
  //   T = column sum of |K|^(2-alpha)
  //   S = row sum of |K|^alpha
  //   loop over prox operators and average coupled entries in T and S
}

void Preconditioner::ComputeEquil() {
  type_ = kPrecondEquil;

  // TODO: implement me
}

void Preconditioner::Renormalize(
    const std::vector<Prox *>& prox_g,
    const std::vector<Prox *>& prox_hc) {

  // TODO: implement me
  // average entries in S and T where prox doesn't allow diagsteps  
  // form matrix M = sqrt(S) * K * sqrt(T)
  // norm=normest(M)
  // T = T / norm
  // S = S / norm
}
