#include "prost/prox/prox_ind_epi_polyhedral.hpp"
#include "prost/prox/vector.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

template<typename T>
inline __device__
void solveLinearSystem2x2(T *A, T *b, T *x)
{
  T determinant = A[0] * A[3] - A[1] * A[2];

  if(determinant != 0)
  {
    x[0] = (b[0] * A[3] - b[1] * A[1]) / determinant;
    x[1] = (b[1] * A[0] - b[0] * A[2]) / determinant;
  }
}

template<typename T>
inline __device__
void solveLinearSystem3x3(T *A, T *b, T *x)
{
  // TODO
}

template<typename T, size_t DIM>
inline __device__
void solveLinearSystem(T *A, T *b, T *x)
{
  switch(DIM)
  {
  case 2:
    solveLinearSystem2x2<T>(A, b, x);
    break;

  case 3:
    // TODO:
    solveLinearSystem3x3<T>(A, b, x);
    break;
  }
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
  const T kAcsTolerance = 5e-6;
  const int kAcsMaxIter = 25;

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
      // compute lhs of inequality constraint and right hand side b
      T lhs = 0;

      rhs[k] = d_coeffs_b[coeff_index + k] + inp_arg[DIM - 1];
      for(int i = 0; i < DIM - 1; i++)
      {
        lhs += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * cur_x[i];
        rhs[k] -= d_coeffs_a[coeff_index + k * (DIM - 1) + i] * inp_arg[i];
      } 

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
        /*
        printf("Iteration %3d: active_set = [", acs_iter);
        for(int k = 0; k < active_set_size; k++)
          printf(" %d ", active_set[k]);
        printf("].\n");
        */

        for(int i = 0; i < DIM; i++)
          prev_x[i] = cur_x[i];

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
            for(int k = 0; k < active_set_size; k++)
            {
              temp[0] += d_coeffs_a[coeff_index + active_set[k] * (DIM - 1)]
                         * rhs[active_set[k]];
              temp[1] -= rhs[active_set[k]];
            }

            solveLinearSystem2x2(mat_ata, temp, cur_x);

            // compute lambda = A_r (A_r^T A_r)^-1 (-x)
            for(int i = 0; i < DIM; i++)
              temp[i] = -cur_x[i];

            solveLinearSystem2x2(mat_ata, temp, temp2);

            for(int k = 0; k < active_set_size; k++)
              cur_lambda[active_set[k]] = d_coeffs_a[coeff_index + active_set[k] * (DIM - 1)]
                                           * temp2[0] - temp2[1];
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

        //
        // check if we are at an optimal point
        //

        // check primal feasibility -> doesn't seem necessary
        /*
        bool primal_feasible = true;
        for(int k = 0; k < coeff_count; k++)
        {
          T Ax = 0;

          for(int i = 0; i < DIM - 1; i++)
            Ax += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * cur_x[i];
          Ax -= cur_x[DIM - 1];

          if(Ax > rhs[k] + kAcsTolerance)
          {
            primal_feasible = false;
            break;
          }
        }

        if(primal_feasible)
        {
          // check dual feasibility
          bool dual_feasible = true;
          for(int k = 0; k < active_set_size; k++)
            if(cur_lambda[active_set[k]] < -kAcsTolerance)
              dual_feasible = false;
          
          if(dual_feasible) // primal & dual feasbile -> finished
          {
            printf("Primal feasible & dual feasible.\n");
            break;
          }
        }
        */

        //
        // determine search direction
        //
        T sum_d = 0;
        for(int i = 0; i < DIM; i++)
        {
          dir[i] = cur_x[i] - prev_x[i];     
          sum_d += abs(dir[i]);
        }

        //printf("sum_d = %f\n", sum_d);

        if(sum_d < kAcsTolerance)
        {
          // 
          // remove most negative lambda from active set
          //
          T best_lambda = -kAcsTolerance;
          int idx_lambda = -1;
          for(int k = 0; k < active_set_size; k++)
          {
            if(cur_lambda[active_set[k]] < best_lambda)
            {
              best_lambda = cur_lambda[active_set[k]];
              idx_lambda = k;
            }
          }

          if(idx_lambda == -1)
          {
            //printf("Primal and dual feasible.\n");
            break;
          }
          else
          {
            //printf("Removing index %d from active set.\n", idx_lambda);
            active_set[idx_lambda] = active_set[active_set_size - 1];
            active_set_size--;
          }          
        }
        else
        {
          //
          // determine smallest t at which a new constraint enters
          // the active set
          //

          T min_step = 1;
          for(int k = 0; k < coeff_count; k++)
          {
            bool in_set = false;
            for(int l = 0; l < active_set_size; l++)
              if(active_set[l] == k)
                in_set = true;

            if(in_set)
              continue;
            
            T Ax = 0, Ad = 0;
            for(int i = 0; i < DIM - 1; i++)
            {
              Ax += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * prev_x[i];
              Ad += d_coeffs_a[coeff_index + k * (DIM - 1) + i] * dir[i];
            }
            Ax -= prev_x[DIM - 1];
            Ad -= dir[DIM - 1];

            T step = (rhs[k] - Ax) / Ad;

            if(Ad >= 0)
              min_step = min(min_step, step);
          }

          //
          // update the primal variable 
          //
          for(int i = 0; i < DIM; i++)
            cur_x[i] = prev_x[i] + min_step * dir[i];

          //
          // determine new active set. TODO: can be optimized by
          // adding blocking constraint
          //
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
          } // for(int k=0; ...)
          
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

  : ProxSeparableSum<T>(index, count, dim, true, false),
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
  static const size_t kBlockSize = 256;

  dim3 block(kBlockSize, 1, 1);
  dim3 grid((this->count_ + block.x - 1) / block.x, 1, 1);

  //std::cout << this->size_ << ", " << this->count_ << ", " << this->dim_ << "." << std::endl;

  size_t shmem_bytes = 5 * 3 * sizeof(T) * kBlockSize;

  std::cout << "Required shared memory: " << shmem_bytes << " bytes." << std::endl;

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