#include "solver/preconditioner.hpp"

#include <iostream>

std::vector<std::vector<int> > GetIndices(const std::vector<Prox<real> *>& prox);
void AverageValues(real *vals, const std::vector<std::vector<int> >& indices);

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

  std::cout << "No preconditioning; estimated matrix norm as " << norm << "." << std::endl;
  
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

void Preconditioner::ComputeAlpha(
    real alpha,
    const std::vector<Prox<real> *>& prox_g,
    const std::vector<Prox<real> *>& prox_hc)
{
  type_ = kPrecondAlpha;

  int m = mat_->nrows();
  int n = mat_->ncols();

  real *h_left = new real[m];
  real *h_right = new real[n];

  std::cout << "m=" << m << ", n=" << n << "...";
  std::cout.flush();

  // compute right preconditioner as sum over matrix columns
  std::cout << "colsum...";
  std::cout.flush();
  for(int i = 0; i < n; i++) {
    real sum = mat_->col_sum(i, 2. - alpha);

    if(sum > 0)
      h_right[i] = 1. / sum;
    else
      h_right[i] = 1.; // should be set to infinity, but might cause NaN
  }
  
  // compute left preconditioner as sum over matrix rows
  std::cout << "rowsum...";
  std::cout.flush();
  for(int i = 0; i < m; i++) {
    real sum = mat_->row_sum(i, alpha);

    if(sum > 0)
      h_left[i] = 1. / sum;
    else
      h_left[i] = 1.; // should be set to infinity, but might cause NaN
  }

  // average diagonal entries where prox doesn't allow diagonal steps
  std::cout << "averaging...";
  std::cout.flush();
  std::vector<std::vector<int> > indices_left = GetIndices(prox_hc);
  std::vector<std::vector<int> > indices_right = GetIndices(prox_g);
  AverageValues(h_left, indices_left);
  AverageValues(h_right, indices_right);
  
  // copy preconditioners to GPU
  cudaMalloc((void **)&d_left_, sizeof(real) * m);
  cudaMalloc((void **)&d_right_, sizeof(real) * n);
  cudaMemcpy(d_left_, h_left, sizeof(real) * m, cudaMemcpyHostToDevice);
  cudaMemcpy(d_right_, h_right, sizeof(real) * n, cudaMemcpyHostToDevice);

  delete [] h_left;
  delete [] h_right;
}

void Preconditioner::ComputeEquil() {
  type_ = kPrecondEquil;

  // TODO: implement me
}


std::vector<std::vector<int> > GetIndices(const std::vector<Prox<real> *>& prox) {
  std::vector<std::vector<int> > indices;
  
  for(int i = 0; i < prox.size(); i++) {
    Prox<real> *p = prox[i];

    if(!p->diagsteps()) {
      for(int j = 0; j < p->count(); j++) {
        int index;
        std::vector<int> inds;

        for(int k = 0; k < p->dim(); k++) {
          if(p->interleaved())
            index = p->index() + j * p->dim() + k;
          else
            index = p->index() + k * p->count() + j;

          inds.push_back(index);
        }

        indices.push_back(inds);        
      }
    }
  }

  return indices;
}

void AverageValues(real *vals, const std::vector<std::vector<int> >& indices) {
  for(int i = 0; i < indices.size(); i++) {
    real avg = 0;
    
    for(int j = 0; j < indices[i].size(); j++) {
      avg += vals[indices[i][j]];
    }

    avg /= (real)indices[i].size();

    for(int j = 0; j < indices[i].size(); j++) {
      vals[indices[i][j]] = avg;
    }
  }
}
