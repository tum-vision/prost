#include "prost/prox/prox_ind_epi_polyhedral.hpp"
#include "prost/prox/vector.hpp"
#include "prost/prox/shared_mem.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"

namespace prost {

struct ShMemCountFun 
{
  inline __host__ __device__
  size_t operator()(size_t dim)
  {
    return dim * dim;
  }
};

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
  size_t count,
  typename ProxIndEpiPolyhedral<T>::InteriorPointParams params)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    Vector<T> res_x(count, DIM-1, true, tx, d_res);
    T& res_y = d_res[count * (DIM-1) + tx];

    const Vector<const T> arg_x(count, DIM-1, true, tx, d_arg);
    const T arg_y = d_arg[count * (DIM-1) + tx];

    size_t coeff_count = d_count[tx];
    size_t coeff_index = d_index[tx];

    // shared memory
    SharedMem<T, ShMemCountFun> sh(DIM, threadIdx.x);
    T *newton_hessian = &sh[0]; // move to array?

    // read coefficients into shared memory

    T current_sol[DIM];
    T newton_step[DIM];
    T newton_gradient[DIM];

#pragma unroll
    for(int i = 0; i < DIM - 1; i++)
      current_sol[i] = arg_x[i];

    current_sol[DIM - 1] = arg_y;

    bool was_feasible = true;

    // generate feasible point for current_sol and check if already feasible
    for(int i = 0; i < coeff_count; i++)
    {
      T ieq = 0;
#pragma unroll
      for(int j = 0; j < DIM - 1; j++)
        ieq += d_coeffs_a[coeff_index + i * (DIM - 1) + j] * current_sol[j];

      ieq += d_coeffs_b[coeff_index + i];

      if(current_sol[DIM - 1] < ieq)
      {
        current_sol[DIM - 1] = ieq;
        was_feasible = false;
      }
    }

    // implement heuristic for initialization

    if(!was_feasible)
    {
      current_sol[DIM - 1] += 1e-2; // make point strictly feasible

      // compute initial penalty parameter
      T sum_a = 0, sum_b = 0;
#pragma unroll
      for(int i = 0; i < DIM - 1; i++)
      {
        sum_a += current_sol[i] - arg_x[i];
      }
      sum_a += current_sol[DIM -  1] - arg_y;

      for(int k = 0; k < coeff_count; k++)
      {
        double factor = -d_coeffs_b[coeff_index + k];

        // compute factor
#pragma unroll
        for(int i = 0; i < DIM - 1; i++) 
        {
          T coeff_a = d_coeffs_a[coeff_index + k * (DIM - 1) + i];
          factor -= coeff_a * current_sol[i];            
        }

        factor += current_sol[DIM - 1];

#pragma unroll
        for(int i = 0; i < DIM - 1; i++)
        {
          T coeff_a = d_coeffs_a[coeff_index + k * (DIM - 1) + i];

          sum_b -= (1 / factor) * coeff_a;
        }

        sum_b += (1 / factor);
      }

      T barrier_t = sum_b / sum_a; 
      //printf("Initial t=%f.\n", barrier_t);

      // project onto polyhedral epigraph using barrier method
      for(int it_b = 0; it_b < params.barrier_max_iter; it_b++) 
      {
        // update current_sol by newton iteration
        for(int it_n = 0; it_n < params.newton_max_iter; it_n++)
        {
          // compute newton step direction:
        
          // init negative gradient to -t(x - \tilde x)
#pragma unroll
          for(int i = 0; i < DIM - 1; i++)
            newton_gradient[i] = -barrier_t * (current_sol[i] - arg_x[i]);

          newton_gradient[DIM - 1] = -barrier_t * (current_sol[DIM - 1] - arg_y);

          // init hessian to t Id
#pragma unroll
          for(int i = 0; i < DIM; i++)
          {

#pragma unroll
            for(int j = 0; j < DIM; j++)
            {
              if(i == j)
                newton_hessian[i + j * DIM] = barrier_t;
              else
                newton_hessian[i + j * DIM] = 0;
            }
          }

          // add terms to negative gradient and hessian
          for(int k = 0; k < coeff_count; k++)
          {
            double factor = -d_coeffs_b[coeff_index + k];

            // compute factor
#pragma unroll
            for(int i = 0; i < DIM - 1; i++) 
            {
              T coeff_a = d_coeffs_a[coeff_index + k * (DIM - 1) + i];
              factor -= coeff_a * current_sol[i];            
            }

            factor += current_sol[DIM - 1];

            factor = 1 / factor;            
            const double factor_sq = factor * factor;

            // TODO: combine with factor calculation -> less read from a
            // update gradient and hessian
#pragma unroll
            for(int i = 0; i < DIM - 1; i++)
            {
              const T coeff_a = d_coeffs_a[coeff_index + k * (DIM - 1) + i];

              newton_gradient[i] -= factor * coeff_a;

#pragma unroll
              for(int j = 0; j < DIM - 1; j++)
              {
                const T coeff_a_j = d_coeffs_a[coeff_index + k * (DIM - 1) + j];

                newton_hessian[i * DIM + j] += factor_sq * coeff_a * coeff_a_j; 
              }

              newton_hessian[(DIM - 1) * DIM + i] += -factor_sq * coeff_a;
              newton_hessian[i * DIM + (DIM - 1)] += -factor_sq * coeff_a;
            }

            newton_gradient[DIM - 1] += factor;
            newton_hessian[DIM * DIM - 1] += factor_sq;
          } // for(int k = 0; ...)

          solveLinearSystem<T, DIM>(newton_hessian, newton_gradient, newton_step);

          // compute lambda^2
          T lambda = 0;
#pragma unroll
          for(int i = 0; i < DIM; i++)
            lambda += newton_gradient[i] * newton_step[i];

          if(lambda / 2 <= params.newton_eps)
            break;

          // optimization: E(x) can be computed only once here.
          // perform line-search to determine newton step size t
          T t = 1;
          double en1, en2;
          bool terminated = false;
          for(int it_l = 0; it_l < params.ls_max_iter; it_l++)
          {
            en1 = 0;  // E(x + t * newton_step)
            en2 = 0;  // E(x)

            T diff;
#pragma unroll
            for(int i = 0; i < DIM - 1; i++)
            {
              diff = current_sol[i] - arg_x[i];
              en2 += 0.5 * diff * diff;

              diff += t * newton_step[i];
              en1 += 0.5 * diff * diff;
            }

            diff = current_sol[DIM - 1] - arg_y;
            en2 += 0.5 * diff * diff;

            diff += t * newton_step[DIM - 1];
            en1 += 0.5 * diff * diff;

            en1 *= barrier_t;
            en2 *= barrier_t;

            for(int k = 0; k < coeff_count; k++)
            {
              T iprod2 = d_coeffs_b[coeff_index + k]; // <(a_i -1), x> + b_i
              T iprod1 = iprod2; // <(a_i -1), x + t * newton_step> + b_i

#pragma unroll
              for(int i = 0; i < DIM - 1; i++)
              {
                const T coeff_a = d_coeffs_a[coeff_index + k * (DIM - 1) + i]; 

                iprod1 += coeff_a * current_sol[i];
                iprod2 += coeff_a * (current_sol[i] + t * newton_step[i]);
              }

              iprod1 -= current_sol[DIM - 1];
              iprod2 -= current_sol[DIM - 1] + t * newton_step[DIM - 1];

              en1 -= log(-iprod1); 
              en2 -= log(-iprod2); 
            }

            if(en1 < en2 - params.ls_alpha * t * lambda) 
            {
              //printf("line-search terminated after %d iters.\n", it_l);
              terminated = true;
              break;
            }

            t *= params.ls_beta;
          }

          if(!terminated)
          {
            //printf("line-search didn't terminate after %d iters! -> increasing penalty parameter\n", 
            //  params.ls_max_iter);
            break;
          }

        //printf("it_barrier=%d, it_newton=%d, linesearch_t=%f, en1=%f, en2=%f, lambda=%f\n", it_b, it_n, t, en1, en2, lambda);
#pragma unroll
          for(int i = 0; i < DIM; i++)
            current_sol[i] += t * newton_step[i];

        }

        double primal_dual_gap = static_cast<double>(coeff_count) / barrier_t;

        if( primal_dual_gap < params.barrier_eps )
        {
          //printf("Reached final tolerance %f. => Converged.\n", static_cast<T>(coeff_count) / barrier_t);
          break;
        }

        barrier_t *= params.barrier_mu; // increase penalty parameter
      }
    } // if(!was_feasible) 

    // write back result
#pragma unroll
    for(int i = 0; i < DIM - 1;i ++)
      res_x[i] = current_sol[i];

    res_y = current_sol[DIM - 1];

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
  const vector<size_t>& index_vec,
  const typename ProxIndEpiPolyhedral<T>::InteriorPointParams& ip_params)

  : ProxSeparableSum<T>(index, count, dim, true, false),
    host_coeffs_a_(coeffs_a), 
    host_coeffs_b_(coeffs_b), 
    host_count_(count_vec), 
    host_index_(index_vec),
    ip_params_(ip_params)
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

  std::cout << this->size_ << ", " << this->count_ << ", " << this->dim_ << "." << std::endl;

  ShMemCountFun fn;
  size_t shmem_bytes = fn(this->dim_) * sizeof(T) * kBlockSize;

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
      this->count_,
      ip_params_
      );
    break;

  case 3:
    ProxIndEpiPolyhedralKernel<T, 3>
    <<<grid, block, shmem_bytes>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(dev_coeffs_a_.data()),
      thrust::raw_pointer_cast(dev_coeffs_b_.data()),
      thrust::raw_pointer_cast(dev_count_.data()),
      thrust::raw_pointer_cast(dev_index_.data()),
      this->count_,
      ip_params_
      );
    break;

  default:
    throw Exception("ProxIndEpiPolyhedral not implemented for dim > 3.");
  }

  std::cout << "Ran kernel.\n";
}

template class ProxIndEpiPolyhedral<float>;
template class ProxIndEpiPolyhedral<double>;

} // namespace prost