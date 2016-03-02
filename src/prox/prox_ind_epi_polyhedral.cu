#include "prost/prox/prox_ind_epi_polyhedral.hpp"
#include "prost/prox/helper.hpp"
#include "prost/prox/vector.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
inline __device__
void solveLinearSystem2x2(T *A, T *b, T *x)
{
  if(abs(A[0]) < abs(A[2]))
  {
    helper::swap(A[2], A[0]);
    helper::swap(A[3], A[1]);
    helper::swap(b[0], b[1]);
  }

  T alpha = A[2] / A[0];
  T beta = A[3] - A[1] * alpha;
  T gamma = b[1] - b[0] * alpha;
  x[1] = gamma / beta;
  x[0] = (b[0] - A[1] * x[1]) / A[0];
}

template<typename T>
inline __device__
void solveLinearSystem3x3(T *A, T *b, T *x) // requires A to be symmetric positive definite.
{
  T d0, d3, d4, d6, d7, d8;
  T y0, y1, y2;

  // compute lower triangular matrix
  d0 = sqrt(A[0]);
  d3 = A[3] / d0;
  d6 = A[6] / d0;
  d4 = sqrt(A[4] - d3 * d3);
  d7 = (A[7] - d6 * d3) / d4;
  d8 = sqrt(A[8] - d6 * d6 - d7 * d7);

  // forward substitution
  y0 = (b[0]) / d0;
  y1 = (b[1] - d3 * y0) / d4;
  y2 = (b[2] - d6 * y0 - d7 * y1) / d8;

  // backward substitution
  x[2] = (y2) / d8;
  x[1] = (y1 - d7 * x[2]) / d4;
  x[0] = (y0 - d3 * x[1] - d6 * x[2]) / d0;
}

template<typename T, int DIM>
inline __device__
void solveLinearSystem(T *A, T *b, T *x) // requires A to be symmetric positive definite.
{
}

template<typename T, size_t DIM>
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
  const int kAcsMaxIter = 250;
  
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    // helper variables
    double dir[3];         // search direction
    double cur_x[3];       // current solution
    double matrix[9];      // matrix
    double temp[3];        // temporary variable
    double inp_arg[3];     // input argument
    double lambda[3];      // lagrange multipliers
    uint8_t active_set[3]; // active set 

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
      double lhs = d_coeffs_a[(coeff_index + k) * (DIM - 1)] * cur_x[0];

      for(int i = 1; i < DIM - 1; i++)
        lhs += d_coeffs_a[(coeff_index + k) * (DIM - 1) + i] * cur_x[i];

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
        int blocking = -1;

        if(active_set_size < DIM) // determine search direction dir 
        {
          if(active_set_size == 1)
          {
            double fac = 1;
            double rhs = -cur_x[DIM - 1] + inp_arg[DIM - 1];
            for(int i = 0; i < DIM - 1; i++)
            {
              const double coeff = d_coeffs_a[(coeff_index + active_set[0]) * (DIM - 1) + i];
              fac += coeff * coeff;
              rhs += coeff * (cur_x[i] - inp_arg[i]);
            }

            // compute \tilde p = A_r^T (A_r A_r^T)^-1 rhs
            for(int i = 0; i < DIM - 1; i++)
              dir[i] = d_coeffs_a[(coeff_index + active_set[0]) * (DIM - 1) + i] * rhs / fac;
            dir[DIM - 1] = -rhs / fac;
          }
          else if(active_set_size == 2)
          {
            // compute A_r A_r^T
            double rhs[2];
            rhs[0] = -cur_x[DIM - 1] + inp_arg[DIM - 1];
            rhs[1] = rhs[0];
            
            matrix[0] = matrix[1] = matrix[3] = 1;
            for(int i = 0; i < DIM - 1; i++)
            {
              const double coeff0 = d_coeffs_a[(coeff_index + active_set[0]) * (DIM - 1) + i];
              const double coeff1 = d_coeffs_a[(coeff_index + active_set[1]) * (DIM - 1) + i];
              matrix[0] += coeff0 * coeff0;
              matrix[1] += coeff0 * coeff1;
              matrix[3] += coeff1 * coeff1;
              
              rhs[0] += coeff0 * (cur_x[i] - inp_arg[i]);
              rhs[1] += coeff1 * (cur_x[i] - inp_arg[i]);
            }
            matrix[2] = matrix[1]; // symmetry

            // compute temp = (A_r A_r^T)^-1 rhs
            solveLinearSystem2x2<double>(matrix, rhs, temp);

            // compute dir = A_r^T temp
            for(int i = 0; i < 2; i++) // active_set_size == 2
              dir[i] = d_coeffs_a[(coeff_index + active_set[0]) * (DIM - 1) + i] * temp[0] +
                       d_coeffs_a[(coeff_index + active_set[1]) * (DIM - 1) + i] * temp[1];
            dir[DIM - 1] = -temp[0] - temp[1];
          }

          // backsubstitution
          double sum_d = 0;
          for(int i = 0; i < DIM; i++)
          {
            dir[i] += -cur_x[i] + inp_arg[i];
            sum_d += abs(dir[i]);
          }

          if(sum_d > kAcsTolerance)
          {
            // determine smallest step size at which a new (blocking) constraint
            // enters the active set
            double min_step = 1;

            for(int k = 0; k < coeff_count; k++)
            {
              // check if constraint k is in active set
              bool in_set = false;
              for(int l = 0; l < active_set_size; l++)
                if(active_set[l] == k)
                {
                  in_set = true;
                  break;
                }
            
              // if so, disregard the constraint
              if(in_set)
                continue;

              double Ax = -cur_x[DIM - 1], Ad = -dir[DIM - 1];
              for(int i = 0; i < DIM - 1; i++)
              {
                const double coeff = d_coeffs_a[(coeff_index + k) * (DIM - 1) + i];
              
                Ax += coeff * cur_x[i];
                Ad += coeff * dir[i];
              }

              if(Ad > 0)
              {
                double step = (d_coeffs_b[coeff_index + k] - Ax) / Ad;
              
                if((step < min_step) && (step > kAcsTolerance))
                {
                  min_step = step;
                  blocking = k;
                }
              }
            }
          
            // update primal variable and add blocking constraint
            for(int i = 0; i < DIM; i++)
              cur_x[i] += min_step * dir[i];
          }
          
          if(blocking != -1) // add blocking constraint to active set
            active_set[active_set_size++] = blocking;
          else if(active_set_size == 1)
          {
            // moved freely without blocking constraint
            // and at least one constraint is active at solution
            // -> converged.
            break;
          }
        }
        
        if(active_set_size == DIM || (blocking == -1))
        {
          if(DIM == 2)
          {
            if(active_set_size == 2)
            {
              // compute A_r^T A_r 
              matrix[0] = 0;
              matrix[1] = 0;
              for(int k = 0; k < active_set_size; k++) 
              {
                const double coeff = d_coeffs_a[(coeff_index + active_set[k]) * (DIM - 1)];
                
                matrix[0] += coeff * coeff;
                matrix[1] -= coeff;
              }
              matrix[2] = matrix[1]; // symmetry
              matrix[3] = DIM; // due to <(-1, ..., -1), (-1, ... -1)>, DIM == active_set_size

              // compute dir = (A_r^T A_r)^-1 -(x + c)
              // dir is used as a temporary variable here.
              for(int i = 0; i < DIM; i++) 
                temp[i] = inp_arg[i] - cur_x[i];
            
              solveLinearSystem2x2<double>(matrix, temp, dir); 
              
              // lambda_r = A_r dir
              for(int k = 0; k < active_set_size; k++) 
                lambda[k] =
                    d_coeffs_a[(coeff_index + active_set[k]) * (DIM - 1)] * dir[0] - dir[1];
            }
            else
              printf("Warning: unusual active set size %d for DIM = 2.\n", active_set_size);
          } 
          else if(DIM == 3)
          {
            if(active_set_size == 2)
            {
              // compute A_r A_r^T
              matrix[0] = matrix[1] = matrix[3] = 1;
              for(int i = 0; i < DIM - 1; i++)
              {
                const double coeff0 = d_coeffs_a[(coeff_index + active_set[0]) * (DIM - 1) + i];
                const double coeff1 = d_coeffs_a[(coeff_index + active_set[1]) * (DIM - 1) + i];
                matrix[0] += coeff0 * coeff0;
                matrix[1] += coeff0 * coeff1;
                matrix[3] += coeff1 * coeff1;
              }
              matrix[2] = matrix[1]; // symmetry

              // compute temp = A_r (-x - c)
              temp[0] = cur_x[DIM - 1] - inp_arg[DIM - 1];
              temp[1] = cur_x[DIM - 1] - inp_arg[DIM - 1];
              
              for(int i = 0; i < DIM - 1; i++) 
              {
                temp[0] +=
                    d_coeffs_a[(coeff_index + active_set[0]) * (DIM - 1) + i] *
                    (inp_arg[i] - cur_x[i]);

                temp[1] +=
                    d_coeffs_a[(coeff_index + active_set[1]) * (DIM - 1) + i] *
                    (inp_arg[i] - cur_x[i]);
              }

              // compute lambda = (A_r A_r^T)^-1 temp
              solveLinearSystem2x2<double>(matrix, temp, lambda);
            }
            else if(active_set_size == 3)
            {
              // TODO: implement pseudo-inverse via SVD.

              matrix[0] = matrix[1] = matrix[2] = matrix[4] = matrix[5] = 0;
              
              // compute A_r^T A_r
              for(int k = 0; k < 3; k++) // active_set_size == 3
              {
                const double coeff0 = d_coeffs_a[(coeff_index + active_set[k]) * (DIM - 1) + 0];
                const double coeff1 = d_coeffs_a[(coeff_index + active_set[k]) * (DIM - 1) + 1];

                matrix[0] += coeff0 * coeff0;
                matrix[1] += coeff1 * coeff0;
                matrix[2] -= coeff0;
                matrix[4] += coeff1 * coeff1;
                matrix[5] -= coeff1;
              }
            
              // symmetry
              matrix[3] = matrix[1];
              matrix[6] = matrix[2];
              matrix[7] = matrix[5];
            
              // due to <(-1, ..., -1), (-1, ... -1)>
              matrix[8] = 3; // active_set_size = 3

              // compute dir = (A_r^T A_r)^-1 (-x - c)
              // dir is used as a temporary variable here.
              for(int i = 0; i < DIM; i++) 
                temp[i] = inp_arg[i] - cur_x[i];

              solveLinearSystem3x3<double>(matrix, temp, dir);
            
              // lambda_r = A_r dir
              for(int k = 0; k < 3; k++) // active_set_size == 3
                lambda[k] =
                    d_coeffs_a[(coeff_index + active_set[k]) * (DIM - 1) + 0] * dir[0] +
                    d_coeffs_a[(coeff_index + active_set[k]) * (DIM - 1) + 1] * dir[1] - dir[2];
            } // else if(active_set_size == 3)
            else
              printf("Warning: unusual active set size %d for DIM = 3.\n", active_set_size);
            
          } // else if(DIM == 3)
     
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
          {
            break;
          }
          else // remove most negative lambda from active set           
            active_set[idx_lambda] = active_set[--active_set_size];
        }        
      } // for(int it_acs = ...)

      if(it_acs == kAcsMaxIter)
        printf("Warning: active set method didn't converge within %d iterations (at tx=%u).\n", kAcsMaxIter, (uint32_t)tx);

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

  if(!(this->dim_ == 2 || this->dim_ == 3))
    throw Exception("Polyhedral epigraph projection only implemented for dimensions 2 and 3.");
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

  case 3:
    ProxIndEpiPolyhedralKernel<T, 3>
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

  default:
    throw Exception("ProxIndEpiPolyhedral not implemented for dim > 3.");
  }
  cudaDeviceSynchronize();

  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    // print the CUDA error message and throw exception
    std::stringstream ss;
    ss << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    throw Exception(ss.str());
  }
}

template class ProxIndEpiPolyhedral<float>;
template class ProxIndEpiPolyhedral<double>;

} // namespace prost

