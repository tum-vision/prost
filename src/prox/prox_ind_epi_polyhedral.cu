#include "prost/prox/prox_ind_epi_polyhedral.hpp"
#include "prost/prox/vector.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

//#define DEBUG_PRINT

template<typename T>
inline __device__
void solveLinearSystem2x2(T *A, T *b, T *x)
{
  if(abs(A[0]) >= abs(A[2]))
  {
    T alpha = A[2] / A[0];
    T beta = A[3] - A[1] * alpha;
    T gamma = b[1] - b[0] * alpha;
    x[1] = gamma / beta;
    x[0] = (b[0] - A[1] * x[1]) / A[0];
  }
  else
  {
    T alpha = A[0] / A[2];
    T beta = A[1] - A[3] * alpha;
    T gamma = b[0] - b[1] * alpha;
    x[1] = gamma / beta;
    x[0] = (b[1] - A[3] * x[1]) / A[2];
  }
}

template<typename T>
inline __device__
void solveLinearSystem3x3(T *A, T *b, T *x)
{
  // TODO
}

template<typename T, size_t DIM, size_t MAX_ACTIVE = DIM>
  __global__
void ProxIndEpiPolyhedralKernel(
    T *d_res,
    const T *d_arg,
    const double *d_coeffs_a,
    const double *d_coeffs_b,
    const uint32_t *d_count, 
    const uint32_t *d_index,
    size_t count,
    bool interleaved)
{
  const double kAcsTolerance = 1e-9;
  const int kAcsMaxIter = 50;
  
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    // helper variables
    double dir[DIM];              // search direction
    double cur_x[DIM];            // current solution
    double inp_arg[DIM];          // input argument
    double matrix[DIM * DIM];     // matrix
    double lambda[MAX_ACTIVE];    // lagrange multipliers
    double temp[DIM];             // temporary variable
    int active_set[MAX_ACTIVE];   // active set

    // read position in coeffs array 
    const uint32_t coeff_count = d_count[tx];
    const uint32_t coeff_index = d_index[tx];

    // read input argument from device memory and initialize current solution to input
    const Vector<const T> arg(count, DIM, interleaved, tx, d_arg);
    for(int i = 0; i < DIM; i++)
      cur_x[i] = inp_arg[i] = arg[i];

    // check feasibility and determine initial active set
    int active_set_size = 0;
    for(int k = 0; k < coeff_count; k++)
    {
      double lhs = d_coeffs_a[coeff_index + k * (DIM - 1)] * cur_x[0];

      for(int i = 1; i < DIM - 1; i++)
        lhs += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * cur_x[i];

      if(lhs - cur_x[DIM - 1] > d_coeffs_b[coeff_index + k] + kAcsTolerance)
      {
        cur_x[DIM - 1] = lhs - d_coeffs_b[coeff_index + k];

        active_set[0] = k;
        active_set_size = 1;
      }
    }

    if(active_set_size > 0) // was not feasible -> run active set method
    {
      int it_acs;
      for(it_acs = 0; it_acs < kAcsMaxIter; it_acs++)
      {

#ifdef DEBUG_PRINT
        printf("Iteration %02d: active_set = [", it_acs);
        for(int k = 0; k < active_set_size; k++)
          printf(" %d ", active_set[k]);
        printf("]. cur_x = [");

        for(int i = 0; i < DIM; i++)
          printf(" %f ", cur_x[i]);
        printf("].\n");
#endif

        if(active_set_size < DIM) // determine search direction dir 
        {
          if(active_set_size == 1)
          {
            double fac = 1;
            double rhs = -cur_x[DIM - 1] + inp_arg[DIM - 1];
            for(int i = 0; i < DIM - 1; i++)
            {
              const double coeff = d_coeffs_a[coeff_index + active_set[0] * (DIM - 1) + i];
              fac += coeff * coeff;
              rhs += coeff * (cur_x[i] - inp_arg[i]);
            }

            // compute \tilde p = A_r^T (A_r A_r^T)^-1 rhs
            for(int i = 0; i < DIM - 1; i++)
              dir[i] = d_coeffs_a[coeff_index + active_set[0] * (DIM - 1) + i] * rhs / fac;
            dir[DIM - 1] = -rhs / fac;

            // backsubstitution
            for(int i = 0; i < DIM; i++)
              dir[i] = dir[i] - cur_x[i] + inp_arg[i];
          }
          else 
            printf("Warning: active set method for DIM >= %d not implemented yet.\n", active_set_size);

#ifdef DEBUG_PRINT
          printf("Found search direction: DIR = [");
          for(int i = 0; i < DIM; i++)
            printf(" %f ", dir[i]);
          printf("].\n");
#endif

          // determine smallest step size at which a new (blocking) constraint
          // enters the active set
          int blocking = -1;
          double min_step = 1;
          for(int k = 0; k < coeff_count; k++)
          {
            bool in_set = false;
            for(int l = 0; l < active_set_size; l++)
              if(active_set[l] == k)
              {
                in_set = true;
                break;
              }
            
            if(in_set)
              continue;
            
            double Ax = -cur_x[DIM - 1], Ad = -dir[DIM - 1];
            for(int i = 0; i < DIM - 1; i++)
            {
              Ax += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * cur_x[i];
              Ad += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * dir[i];
            }

            if(Ad > 0)
            {
              double step = (d_coeffs_b[coeff_index + k] - Ax) / Ad;

              if(step < min_step)
              {
                min_step = step;
                blocking = k;
              }
            }
          }

          // update primal variable and add blocking constraint
          for(int i = 0; i < DIM; i++)
            cur_x[i] = cur_x[i] + min_step * dir[i];
          
          if(blocking != -1)
          {
            active_set[active_set_size] = blocking;
            active_set_size++;
          }
          else // moved without finding blocking constraint -> converged!
            break;
        }
        else // active_set has size DIM -> converged or throw out constraint.
        {
          if(DIM == 2)
          {
            // compute A_r^T A_r 
            matrix[0] = 0;
            matrix[1] = 0;
            for(int k = 0; k < active_set_size; k++)
            {
              const double coeff = d_coeffs_a[coeff_index + active_set[k] * (DIM - 1)];
              
              matrix[0] += coeff * coeff;
              matrix[1] -= coeff;
            }
            matrix[2] = matrix[1]; // symmetry
            matrix[3] = active_set_size; // due to <(-1, ..., -1), (-1, ... -1)>

            // compute dir = (A_r^T A_r)^-1 (x + c)
            // dir is used as a temporary variable here.
            for(int i = 0; i < DIM; i++) 
              temp[i] = -(cur_x[i] - inp_arg[i]);
            
            solveLinearSystem2x2<double>(matrix, temp, dir); 
              
            // lambda_r = A_r dir
            for(int k = 0; k < active_set_size; k++) 
              lambda[k] = d_coeffs_a[coeff_index + active_set[k] * (DIM - 1) + 0] * dir[0] - dir[1];
          }
          else
            printf("Warning: active set method for DIM > 2 not implemented yet.\n");
     
          double best_lambda = 0;
          int idx_lambda = -1;
          for(int k = 0; k < active_set_size; k++)
          {
            if(lambda[k] < best_lambda)
            {
              best_lambda = lambda[k];
              idx_lambda = k;
            }
          }
          
          // if all lambda >= 0 -> solution
          if(idx_lambda == -1)
            break;
          else           
          {
            // remove most negative lambda from active set
            active_set[idx_lambda] = active_set[active_set_size - 1];
            active_set_size--;

#ifdef DEBUG_PRINT
            printf("Removing constraint %d from active set as it has a negative Lagrange multiplier with lambda = %f..\n", active_set[idx_lambda], best_lambda);
#endif
          }
        }
      } // for(int it_acs = ...)

      if(it_acs == kAcsMaxIter)
        printf("Warning: active set method didn't converge within %d iterations.\n", kAcsMaxIter);

    } // if(active_set_size > 0)

    // write out result
    Vector<T> res(count, DIM, interleaved, tx, d_res);
    for(int i = 0; i < DIM; i++)
      res[i] = cur_x[i];

  } // if(tx < count)
}

template<typename T>
ProxIndEpiPolyhedral<T>::ProxIndEpiPolyhedral(
  size_t index,
  size_t count,
  size_t dim, 
  bool interleaved,
  const vector<double>& coeffs_a,
  const vector<double>& coeffs_b, 
  const vector<uint32_t>& count_vec,
  const vector<uint32_t>& index_vec)

  : ProxSeparableSum<T>(index, count, dim, interleaved, false),
    host_coeffs_a_(coeffs_a), 
    host_coeffs_b_(coeffs_b), 
    host_count_(count_vec), 
    host_index_(index_vec)
{
}

template<typename T>
void ProxIndEpiPolyhedral<T>::Initialize()
{
  if(host_coeffs_a_.empty() ||
     host_coeffs_b_.empty() ||
     host_count_.empty() ||
     host_index_.empty())
  {
    throw Exception("ProxIndEpiPolyhedral: empty data array!");
  }

  if(host_index_.size() != this->count() || host_count_.size() != this->count())
    throw Exception("Count doesn't match size of indices/counts array!");

  // copy and allocate data on GPU
  try 
  {
    dev_coeffs_a_ = host_coeffs_a_;
    dev_coeffs_b_ = host_coeffs_b_;
    dev_count_ = host_count_;
    dev_index_ = host_index_;
  }
  catch(std::bad_alloc& e) 
  {
    throw Exception("Out of memory.");
  }
}

template<typename T>
size_t ProxIndEpiPolyhedral<T>::gpu_mem_amount() const
{
  return (host_coeffs_a_.size() + host_coeffs_b_.size()) * sizeof(double) + 
    (host_count_.size() + host_index_.size()) * sizeof(uint32_t);
}

template<typename T>
void ProxIndEpiPolyhedral<T>::EvalLocal(
  const typename device_vector<T>::iterator& result_beg,
  const typename device_vector<T>::iterator& result_end,
  const typename device_vector<T>::const_iterator& arg_beg,
  const typename device_vector<T>::const_iterator& arg_end,
  const typename device_vector<T>::const_iterator& tau_beg,
  const typename device_vector<T>::const_iterator& tau_end,
  T tau,
  bool invert_tau)
{
  dim3 block(kBlockSizeCUDA, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  //std::cout << this->size_ << ", " << this->count_ << ", " << this->dim_ << "." << std::endl;

  //size_t shmem_bytes = 10 * 3 * sizeof(T) * kBlockSize;

  //std::cout << "Required shared memory: " << shmem_bytes << " bytes." << std::endl;

  // TODO: warm-start with previous solution?

  switch(this->dim_)
  {
  case 2:
    ProxIndEpiPolyhedralKernel<T, 2>
    <<<grid, block>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(dev_coeffs_a_.data()),
      thrust::raw_pointer_cast(dev_coeffs_b_.data()),
      thrust::raw_pointer_cast(dev_count_.data()),
      thrust::raw_pointer_cast(dev_index_.data()),
      this->count_,
      this->interleaved_
      );
    break;
/*
  case 3:
    ProxIndEpiPolyhedralKernel<T, 3>
    <<<grid, block, shmem_bytes>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(dev_coeffs_a_.data()),
      thrust::raw_pointer_cast(dev_coeffs_b_.data()),
      thrust::raw_pointer_cast(dev_count_.data()),
      thrust::raw_pointer_cast(dev_index_.data()),
      this->count_
      );
    break;
*/
  default:
    throw Exception("ProxIndEpiPolyhedral not implemented for dim > 2.");
  }
  cudaDeviceSynchronize();
}

template class ProxIndEpiPolyhedral<float>;
template class ProxIndEpiPolyhedral<double>;

} // namespace prost