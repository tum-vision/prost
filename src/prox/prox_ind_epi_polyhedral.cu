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
    return 4 * dim + 2 * dim * dim;
  }
};

template<typename T>
__inline__ __device__ void Cholesky(int d, T* S, T* D)
{
  for(int k=0;k<d;++k){
    double sum=0.;
    for(int p=0;p<k;++p)sum+=D[k*d+p]*D[k*d+p];
    D[k*d+k]=sqrt(S[k*d+k]-sum);
    for(int i=k+1;i<d;++i){
      double sum=0.;
      for(int p=0;p<k;++p)sum+=D[i*d+p]*D[k*d+p];
      D[i*d+k]=(S[i*d+k]-sum)/D[k*d+k];
    }
  }
}

template<typename T>
__inline__ __device__
void solveCholesky(int d,T* LU, T* b, T* x, T* y)
{
   for(int i=0;i<d;++i){
      double sum=0.;
      for(int k=0;k<i;++k)sum+=LU[i*d+k]*y[k];
      y[i]=(b[i]-sum)/LU[i*d+i];
   }
   for(int i=d-1;i>=0;--i){
      double sum=0.;
      for(int k=i+1;k<d;++k)sum+=LU[k*d+i]*x[k];
      x[i]=(y[i]-sum)/LU[i*d+i];
   }
}

// dim as template
template<typename T>
__global__
void ProxIndEpiPolyhedralKernel(
  T *d_res,
  const T *d_arg,
  const T *d_coeffs_a,
  const T *d_coeffs_b,
  const size_t *d_count,
  const size_t *d_index,
  size_t count,
  size_t dim)
{
  // TODO: optimize over hyperparameters (grid-search)
  T barrier_t = 10;
  const T barrier_mu = 3;
  const T barrier_eps = 2e-6;
  const int barrier_max_iter = 100;

  const T newton_eps = 1e-5;
  const T newton_alpha = 0.2;
  const T newton_beta = 0.8;
  const int newton_max_iter = 25;
  const int line_search_max_iter = 10;

  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    Vector<T> res_x(count, dim-1, true, tx, d_res);
    T& res_y = d_res[count * (dim-1) + tx];

    const Vector<const T> arg_x(count, dim-1, true, tx, d_arg);
    const T arg_y = d_arg[count * (dim-1) + tx];

    size_t coeff_count = d_count[tx];
    size_t coeff_index = d_index[tx];

    SharedMem<T, ShMemCountFun> sh(dim, threadIdx.x);
    T *current_sol = &sh[0]; // move to array
    T *newton_step = &sh[dim];
    T *newton_gradient = &sh[2 * dim];
    T *temp_array = &sh[3 * dim]; // no need -> overwrite gradient and store in local array
    T *newton_hessian = &sh[4 * dim];
    T *newton_LU = &sh[4 * dim + dim * dim];

    // initialize current_sol with input argument
    for(int i = 0; i < dim - 1; i++)
      current_sol[i] = arg_x[i];

    current_sol[dim - 1] = arg_y;

    bool was_feasible = true;
    // generate feasible point for current_sol and check if already feasible
    for(int i = 0; i < coeff_count; i++)
    {
      T ieq = 0;
      for(int j = 0; j < dim - 1; j++)
        ieq += d_coeffs_a[coeff_index + i * (dim - 1) + j] * current_sol[j];

      ieq += d_coeffs_b[coeff_index + i];

      if(current_sol[dim - 1] < ieq)
      {
        current_sol[dim - 1] = ieq;
        was_feasible = false;
      }
    }

    if(!was_feasible)
    {
      current_sol[dim - 1] += 10;

      // project onto polyhedral epigraph using barrier method
      for(int it_b = 0; it_b < barrier_max_iter; it_b++) 
      {
        // update current_sol by newton iteration
        for(int it_n = 0; it_n < newton_max_iter; it_n++)
        {
          // compute newton step direction:
        
          // init negative gradient to -t(x - \tilde x)
          for(int i = 0; i < dim - 1; i++)
            newton_gradient[i] = -barrier_t * (current_sol[i] - arg_x[i]);

          newton_gradient[dim - 1] = -barrier_t * (current_sol[dim - 1] - arg_y);

          // init hessian to t Id
          for(int i = 0; i < dim; i++)
            for(int j = 0; j < dim; j++)
              if(i == j)
                newton_hessian[i + j * dim] = barrier_t;
              else
                newton_hessian[i + j * dim] = 0;

          // add terms to negative gradient and hessian
          for(int k = 0; k < coeff_count; k++)
          {
            double factor = -d_coeffs_b[coeff_index + k];

            // compute factor
            for(int i = 0; i < dim - 1; i++) 
            {
              T coeff_a = d_coeffs_a[coeff_index + k * (dim - 1) + i];
              factor -= coeff_a * current_sol[i];            
            }

            factor += current_sol[dim - 1];
/*
            if(abs(factor) < 1e-6)
              factor = factor > 0 ? 1e-6 : -1e-6;
*/

            factor = 1 / factor;            
            const double factor_sq = factor * factor;

            // TODO: combine with factor calculation -> less read from a
            // update gradient and hessian
            for(int i = 0; i < dim - 1; i++)
            {
              const T coeff_a = d_coeffs_a[coeff_index + k * (dim - 1) + i];

              newton_gradient[i] -= factor * coeff_a;

              for(int j = 0; j < dim - 1; j++)
              {
                const T coeff_a_j = d_coeffs_a[coeff_index + k * (dim - 1) + j];

                newton_hessian[i * dim + j] += factor_sq * coeff_a * coeff_a_j; 
              }

              newton_hessian[(dim - 1) * dim + i] += -factor_sq * coeff_a;
              newton_hessian[i * dim + (dim - 1)] += -factor_sq * coeff_a;
            }

            newton_gradient[dim - 1] += factor;
            newton_hessian[dim * dim - 1] += factor_sq;
          } // for(int k = 0; ...)

          // Cholesky-factorize hessian
          Cholesky(dim, newton_hessian, newton_LU);

          // compute newton_step 
          solveCholesky(dim, newton_LU, newton_gradient, newton_step, temp_array);

          // compute lambda^2
          T lambda = 0;
          for(int i = 0; i < dim; i++)
            lambda += newton_gradient[i] * newton_step[i];

          if(lambda / 2 <= newton_eps)
            break;

          // optimization: E(x) can be computed only once here.
          // perform line-search to determine newton step size t
          T t = 1;
          double en1, en2;
          bool terminated = false;
          for(int it_l = 0; it_l < line_search_max_iter; it_l++)
          {
            en1 = 0;  // E(x + t * newton_step)
            en2 = 0;  // E(x)

            T diff;
            for(int i = 0; i < dim - 1; i++)
            {
              diff = current_sol[i] - arg_x[i];
              en2 += 0.5 * diff * diff;

              diff += t * newton_step[i];
              en1 += 0.5 * diff * diff;
            }

            diff = current_sol[dim - 1] - arg_y;
            en2 += 0.5 * diff * diff;

            diff += t * newton_step[dim - 1];
            en1 += 0.5 * diff * diff;

            en1 *= barrier_t;
            en2 *= barrier_t;

            for(int k = 0; k < coeff_count; k++)
            {
              T iprod2 = d_coeffs_b[coeff_index + k]; // <(a_i -1), x> + b_i
              T iprod1 = iprod2; // <(a_i -1), x + t * newton_step> + b_i

              for(int i = 0; i < dim - 1; i++)
              {
                const T coeff_a = d_coeffs_a[coeff_index + k * (dim - 1) + i]; 

                iprod1 += coeff_a * current_sol[i];
                iprod2 += coeff_a * (current_sol[i] + t * newton_step[i]);
              }

              iprod1 -= current_sol[dim - 1];
              iprod2 -= current_sol[dim - 1] + t * newton_step[dim - 1];
/*
              if(iprod1 > -1e-6 || iprod2 > -1e-6) 
              {
                en1 = 1e10;
                en2 = 0;
                break;
              }
*/

              en1 -= log(-iprod1); 
              en2 -= log(-iprod2); 
            }

            if(en1 < en2 - newton_alpha * t * lambda) 
            {
              //printf("line-search terminated after %d iters.\n", it_l);
              terminated = true;
              break;
            }

            t *= newton_beta;
          }

          if(!terminated)
          {
            //printf("line-search didn't terminate! -> increasing penalty parameter\n");
            break;
          }

          //printf("it_barrier=%d, it_newton=%d, linesearch_t=%f, en1=%f, en2=%f, lambda=%f\n", it_b, it_n, t, en1, en2, lambda);
          for(int i = 0; i < dim; i++)
            current_sol[i] += t * newton_step[i];

        }

        if( static_cast<T>(coeff_count) / barrier_t < barrier_eps )
          break;

        barrier_t *= barrier_mu; // increase penalty parameter
      }
    } // if(!was_feasible) 

    // write back result
    for(int i = 0; i < dim - 1;i ++)
      res_x[i] = current_sol[i];

    res_y = current_sol[dim - 1];

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

  std::cout << this->size_ << ", " << this->count_ << ", " << this->dim_ << "." << std::endl;

  ShMemCountFun fn;
  size_t shmem_bytes = fn(this->dim_) * sizeof(T) * kBlockSize;

  std::cout << "Required shared memory: " << shmem_bytes << " bytes." << std::endl;

  // TODO: warm-start with previous solution
  ProxIndEpiPolyhedralKernel<T>
    <<<grid, block, shmem_bytes>>>(
      thrust::raw_pointer_cast(&(*result_beg)),
      thrust::raw_pointer_cast(&(*arg_beg)),
      thrust::raw_pointer_cast(dev_coeffs_a_.data()),
      thrust::raw_pointer_cast(dev_coeffs_b_.data()),
      thrust::raw_pointer_cast(dev_count_.data()),
      thrust::raw_pointer_cast(dev_index_.data()),
      this->count_,
      this->dim_);
}

template class ProxIndEpiPolyhedral<float>;
template class ProxIndEpiPolyhedral<double>;

} // namespace prost