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
void solveLinearSystem2x2Precise(T *A, T *b, T *x)
{
  solveLinearSystem2x2<T>(A, b, x);

  T db[2], dx[2];
  db[0] = A[0] * x[0] + A[1] * x[1] - b[0];
  db[1] = A[2] * x[0] + A[3] * x[1] - b[1];
  solveLinearSystem2x2<T>(A, db, dx);

  x[0] -= dx[0];
  x[1] -= dx[1];
}

template<typename T>
inline __device__
void solveLinearSystem3x3(T *A, T *b, T *x)
{
  // TODO
}

template<typename T, size_t DIM>
__global__
void ProxIndEpiPolyhedralKernel(
  T *d_res,
  const T *d_arg,
  const T *d_coeffs_a,
  const T *d_coeffs_b,
  const size_t *d_count,
  const size_t *d_index,
  size_t count)
{
  const T kAcsTolerance = 1e-6;
  const int kAcsMaxIter = 20;

  // get pointer to shared memory
  extern __shared__ char sh_mem[];

  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    //
    // helper variables
    //
    T dir[DIM];              // search direction
    T prev_x[DIM];           // previous solution
    T cur_x[DIM];            // current solution x^k
    T inp_arg[DIM];          // input argument
    T mat_ata[DIM * DIM];    // A^T A
    T temp[DIM], temp2[DIM]; // temporary variables

    //
    // read input argument
    //
    const Vector<const T> arg_x(count, DIM-1, true, tx, d_arg);
    const T arg_y = d_arg[count * (DIM-1) + tx];
    for(int i = 0; i < DIM - 1; i++)
      inp_arg[i] = arg_x[i];
    inp_arg[DIM - 1] = arg_y;

    //
    // read position in coeffs array 
    //
    const size_t coeff_count = d_count[tx];
    const size_t coeff_index = d_index[tx];

    //
    // set up variables in shared memory
    //
    const int size_in_bytes = 3 * sizeof(T);
    const size_t sh_pos = (coeff_index - d_index[blockDim.x * blockIdx.x]) * size_in_bytes; 
    const size_t sh_pos_lambda = sh_pos + coeff_count * sizeof(T);
    const size_t sh_pos_active = sh_pos + coeff_count * 2 * sizeof(T);

    T *rhs = reinterpret_cast<T*>(&sh_mem[sh_pos]);
    T *cur_lambda = reinterpret_cast<T*>(&sh_mem[sh_pos_lambda]);
    int *active_set = reinterpret_cast<int *>(&sh_mem[sh_pos_active]);
    int active_set_size;

    //
    // initialize current solution to input argument and check feasibility
    //
    bool was_feasible = true;
    for(int i = 0; i < DIM; i++)
      cur_x[i] = 0;

    for(int k = 0; k < coeff_count; k++)
    {
      // compute right hand side b
      rhs[k] = d_coeffs_b[coeff_index + k] + inp_arg[DIM - 1];
      for(int i = 0; i < DIM - 1; i++)
        rhs[k] -= d_coeffs_a[coeff_index + k * (DIM - 1) + i] * inp_arg[i];
    }

    for(int k = 0; k < coeff_count; k++)
    {
      T lhs = 0;

      for(int i = 0; i < DIM - 1; i++)
        lhs += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * cur_x[i];

      // check if current solution is feasible w.r.t. inequality constraint
      if(lhs - cur_x[DIM - 1] > rhs[k])
      {
        // make solution feasible
        cur_x[DIM - 1] = lhs - rhs[k];
        was_feasible = false;
      }
    }

    //
    // if the initial solution was not feasible, use active set method to find projection
    //
    if(!was_feasible)
    {
      // determine initial active set
      active_set_size = 0;
      for(int k = 0; k < coeff_count; k++)
      {
        T ieq = 0;
        for(int i = 0; i < DIM - 1; i++)
          ieq += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * cur_x[i];
        ieq -= cur_x[DIM - 1];

        if( abs(ieq - rhs[k]) < kAcsTolerance )
        {
          active_set[active_set_size] = k;
          active_set_size++;
        }
      }

      // run active set method
      for(int acs_iter = 0; acs_iter < kAcsMaxIter; acs_iter++)
      {
#ifdef DEBUG_PRINT
        printf("Iteration %3d: active_set = [", acs_iter);
        for(int k = 0; k < active_set_size; k++)
          printf(" %d ", active_set[k]);
        printf("]. old_x = [");
#endif

        for(int i = 0; i < DIM; i++)
        {
#ifdef DEBUG_PRINT
          printf(" %f ", cur_x[i]);
#endif
          prev_x[i] = cur_x[i];
        }
#ifdef DEBUG_PRINT
        printf("]. ");
#endif

        //
        // solve equality constrained system with current active set
        //
        if(DIM == 2)
        {
          if(active_set_size == 1)
          {
            // compute A_r A_r^T
            T fac = 1;
            for(int i = 0; i < DIM - 1; i++)
            {
              const T coeff = d_coeffs_a[coeff_index + active_set[0] * (DIM - 1) + i];
              fac += coeff * coeff;
            }

            // compute x = A_r^T (A_r A_r^T)^-1 b_r
            for(int i = 0; i < DIM - 1; i++)
              cur_x[i] = d_coeffs_a[coeff_index + active_set[0] * (DIM - 1) + i] * rhs[active_set[0]] / fac;
            cur_x[DIM - 1] = -rhs[active_set[0]] / fac;

            // compute lambda = (A_r A_r^T)^-1 (-A_r x)
            T Ax = -cur_x[DIM - 1];
            for(int i = 0; i < DIM - 1; i++)
              Ax += d_coeffs_a[coeff_index + active_set[0] * (DIM - 1) + i] * cur_x[i];

            cur_lambda[active_set[0]] = -Ax / fac;
          }
          else if(active_set_size > 1)
          {
            // compute A_r^T A_r 
            mat_ata[0] = 0;
            mat_ata[1] = 0;
            for(int k = 0; k < active_set_size; k++)
            {
              const T coeff = d_coeffs_a[coeff_index + active_set[k] * (DIM - 1)];

              mat_ata[0] += coeff * coeff;
              mat_ata[1] -= coeff;
            }
            mat_ata[2] = mat_ata[1]; // symmetry
            mat_ata[3] = active_set_size; // due to <(-1, ..., -1), (-1, ... -1)>

            // compute x = (A_r^T A_r)^-1 A_r^T b_r
            temp[0] = 0; temp[1] = 0;
            for(int k = 0; k < active_set_size; k++) // A_r^T b_r
            {
              temp[0] += d_coeffs_a[coeff_index + active_set[k] * (DIM - 1) + 0]
                * rhs[active_set[k]];
              temp[1] -= rhs[active_set[k]];
            }

            solveLinearSystem2x2Precise<T>(mat_ata, temp, cur_x);

            // compute lambda_r = A_r (A_r^T A_r)^-1 (-x)
            for(int i = 0; i < DIM; i++) // temp = -x
              temp[i] = -cur_x[i];

            solveLinearSystem2x2Precise<T>(mat_ata, temp, temp2); // temp2 = (A_r^T A_r)^-1 temp

            for(int k = 0; k < active_set_size; k++) // lambda_r = A_r temp2
              cur_lambda[active_set[k]] = d_coeffs_a[coeff_index + active_set[k] * (DIM - 1) + 0]
                                           * temp2[0] - temp2[1];

            for(int i = 0; i < DIM - 1; i++)
              prev_x[i] = cur_x[i];
          }
        }
        else if(DIM == 3)
        {
          // TODO: implement for DIM == 3

          if(active_set_size == 1)
          {
          }
          else if(active_set_size == 2)
          {
          }
          else if(active_set_size > 2)
          {
          }
        }

#ifdef DEBUG_PRINT
        printf("potential_x = [ ");
        for(int i = 0; i < DIM; i++)
        {
          printf(" %f ", cur_x[i]);
        }
        printf("]. ");
#endif

        //
        // determine search direction
        //
        T norm_d = 0;
        for(int i = 0; i < DIM; i++)
        {
          dir[i] = cur_x[i] - prev_x[i];     
          norm_d += dir[i] * dir[i];
        }
        norm_d = sqrt(norm_d);

        if(norm_d > kAcsTolerance)
          for(int i = 0; i < DIM; i++)
            dir[i] /= norm_d;

#ifdef DEBUG_PRINT
        printf("search_dir = [ ");
        for(int i = 0; i < DIM; i++)
        {
          printf(" %f ", dir[i]);
        }
        printf("]. \n");

        printf("norm_d = %f\n", norm_d);
#endif

        if(norm_d < kAcsTolerance)
        {
          // 
          // remove most negative lambda from active set
          //
          T best_lambda = 0;
          int idx_lambda = -1;
          for(int k = 0; k < active_set_size; k++)
          {
            if(cur_lambda[active_set[k]] < best_lambda)
            {
              best_lambda = cur_lambda[active_set[k]];
              idx_lambda = k;
            }
          }

          //
          // check if we are at an optimal point
          //
          if(idx_lambda == -1)
          {
#ifdef DEBUG_PRINT
            printf("Primal and dual feasible.\n");
#endif
            break;
          }
          else
          {
#ifdef DEBUG_PRINT
            printf("Removing constraint %d from active set as it has a negative Lagrange multiplier with lambda = %f..\n", active_set[idx_lambda], best_lambda);
#endif
            active_set[idx_lambda] = active_set[active_set_size - 1];
            active_set_size--;
          }          
        }
        else
        {
          //
          // determine smallest step size at which a new (blocking) constraint
          // enters the active set
          //
          int blocking = -1;
          T min_step = norm_d;
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
            
            T Ax = -prev_x[DIM - 1], Ad = -dir[DIM - 1];
            for(int i = 0; i < DIM - 1; i++)
            {
              Ax += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * prev_x[i];
              Ad += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * dir[i];
            }

            if(Ad > 0)
            {
              T step = (rhs[k] - Ax) / Ad;
              
              if((step < min_step) && (step > kAcsTolerance))
              {
                min_step = step;
                blocking = k;
              }
            }
          }

#ifdef DEBUG_PRINT
          printf("Found step size %f with blocking constraint %d.\n", min_step, blocking);
#endif

          //
          // update the primal variable 
          //
          for(int i = 0; i < DIM; i++)
            cur_x[i] = prev_x[i] + min_step * dir[i];

          //
          // determine new active set by adding blocking constraint
          //
          if(blocking != -1)
          {
            active_set[active_set_size] = blocking;
            active_set_size++;
          }
          
        } // else
      } // for(int acs_iter = 0; ...)

    } // if(!was_feasible) 

    //
    // write back result
    //
    Vector<T> res_x(count, DIM-1, true, tx, d_res);
    T& res_y = d_res[count * (DIM-1) + tx];

    for(int i = 0; i < DIM - 1;i ++)
      res_x[i] = cur_x[i] + inp_arg[i];

    res_y = cur_x[DIM - 1] + inp_arg[DIM - 1];
  } // if(tx < count)
}

template<typename T>
ProxIndEpiPolyhedral<T>::ProxIndEpiPolyhedral(
  size_t index,
  size_t count,
  size_t dim, 
  const vector<T>& coeffs_a,
  const vector<T>& coeffs_b, 
  const vector<size_t>& count_vec,
  const vector<size_t>& index_vec)

  : ProxSeparableSum<T>(index, count, dim, false, false),
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
  return (host_coeffs_a_.size() + host_coeffs_b_.size()) * sizeof(T) + 
    (host_count_.size() + host_index_.size()) * sizeof(size_t);
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
  static const size_t kBlockSize = 128;

  dim3 block(kBlockSize, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  //std::cout << this->size_ << ", " << this->count_ << ", " << this->dim_ << "." << std::endl;

  size_t shmem_bytes = 10 * 3 * sizeof(T) * kBlockSize;

  //std::cout << "Required shared memory: " << shmem_bytes << " bytes." << std::endl;

  // TODO: warm-start with previous solution?

  switch(this->dim_)
  {
  case 2:
    ProxIndEpiPolyhedralKernel<T, 2>
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