#include "prost/prox/prox_ind_epi_polyhedral_1d.hpp"
#include "prost/config.hpp"
#include "prost/exception.hpp"
#include "prost/math_functions.hpp"

namespace prost {

template<typename T>
__global__
void ProxIndEpiPolyhedral1DKernel(
  T *d_res,
  const T *d_arg,
  const T *d_alpha,
  const T *d_beta,
  const T *d_pt_x,
  const T *d_pt_y,
  const size_t *d_count,
  const size_t *d_index,
  size_t count,
  bool interleaved)
{
  size_t tx = threadIdx.x + blockDim.x * blockIdx.x;

  if(tx < count)
  {
    T result[2];

    // get v = (x0, y0) and alpha,beta and count,index
    T alpha = d_alpha[tx];
    T beta = d_beta[tx];
    size_t count_local = d_count[tx];
    size_t index = d_index[tx];

    T v[2];
    if(interleaved)
    {
      v[0] = d_arg[tx * 2 + 0];
      v[1] = d_arg[tx * 2 + 1];
    }
    else
    {
      v[0] = d_arg[tx + count * 0];
      v[1] = d_arg[tx + count * 1];
    }
    
    // compute vector normal to slope for feasibility-check
    T n_slope[2];
    n_slope[0] = alpha;
    n_slope[1] = -1;
    
    T x1 = d_pt_x[index];
    T y1 = d_pt_y[index];
    T p[2];
    p[0] = x1;
    p[1] = y1;

    bool feasible_left = PointInHalfspace(v, p, n_slope, 2);
    
    T n_halfspace[2];
    n_halfspace[0] = 1;
    n_halfspace[1] = alpha;

    bool halfspace_left = PointInHalfspace(v, p, n_halfspace, 2);

    bool projected = false;

    if(!feasible_left && halfspace_left)
    {
      // point is not feasible wrt to 0-th piece and
      //  lies in rectangle => projection is the 
      //  respective half space projection
      T t = x1*n_slope[0] + y1*n_slope[1];
      ProjectHalfspace(v, n_slope, t, result, 2);
      projected = true;
    }

    if(!projected)
    {
      for(size_t i = 0; i < count_local-1; i++)
      {
        // read "kink" at i+1
        T x2 = d_pt_x[index+i+1];
        T y2 = d_pt_y[index+i+1];

        // compute slope
        T c = (y2-y1) / (x2-x1);

        // compute vector normal to slope
        n_slope[0] = c;
        n_slope[1] = -1;

        // check whether point v is feasible wrt i-th piece
        bool feasible_right = PointInHalfspace(v, p, n_slope, 2);

        n_halfspace[0] = -1;
        n_halfspace[1] = -c;

        bool halfspace_right = PointInHalfspace(v, p, n_halfspace, 2);

        p[0] = x2;
        p[1] = y2;
        if(!feasible_left || !feasible_right)
	{
          // point is not feasible wrt to i-th piece or (i-1)-th piece
          if(!halfspace_left && !halfspace_right)
	  {
            // point lies in (i-1)-th normal cone => projection is onto the "kink"
            result[0] = x1;
            result[1] = y1; 

            projected = true;
            break;
          }

          // compute inverse normal -n s.t. the two normals n and -n
          //  together with the two kinks define a rectangle
          n_halfspace[0] = -n_halfspace[0];
          n_halfspace[1] = -n_halfspace[1];

          // check wether phoint lies in i-th halfspace
          halfspace_left = PointInHalfspace(v, p, n_halfspace, 2);
          if(halfspace_right && halfspace_left)
	  {
            // point lies in i-th rectangle => projection is the 
            //  respective half space projection
            T t = x1*n_slope[0] + y1*n_slope[1];
            ProjectHalfspace(v, n_slope, t, result, 2);

            projected = true;
            break;
          }
        }

        // hand over variables for next iteration
        x1 = x2;
        y1 = y2;
        feasible_left = feasible_right;
      }
    }

    if(!projected)
    {
      // compute vector normal to slope
      n_slope[0] = beta;
      n_slope[1] = -1; 

      // check whether point v is feasible wrt i-th piece
      bool feasible_right = PointInHalfspace(v, p, n_slope, 2);

      n_halfspace[0] = -1;
      n_halfspace[1] = -beta;

      bool halfspace_right = PointInHalfspace(v, p, n_halfspace, 2);

      if(!feasible_left || !feasible_right)
      {
        // point is not feasible wrt to i-th piece or (i-1)-th piece
        if(!halfspace_left && !halfspace_right)
	{
          // point lies in last normal cone => projection is the last "kink"
          result[0] = x1;
          result[1] = y1; 

          projected = true;
        }
	else if(halfspace_right)
	{
          // point lies in last rectangle => projection is the 
          //  respective half space projection
          T t = x1*n_slope[0] + y1*n_slope[1];
          ProjectHalfspace(v, n_slope, t, result, 2);

          projected = true;
        }
      }
    }

    // point has not been projected. That means we output the original point    
    if(!projected)
    {
      result[0] = v[0];
      result[1] = v[1];      
    }
    
    // write out result
    if(interleaved)
    {
      d_res[tx * 2 + 0] = result[0];
      d_res[tx * 2 + 1] = result[1];
    }
    else
    {
      d_res[tx + count * 0] = result[0];
      d_res[tx + count * 1] = result[1];
    }
  }
}
  
template<typename T>
ProxIndEpiPolyhedral1D<T>::ProxIndEpiPolyhedral1D(
  size_t index,
  size_t count,
  bool interleaved,
  const vector<T>& pt_x, 
  const vector<T>& pt_y,
  const vector<T>& alpha,
  const vector<T>& beta,
  const vector<size_t>& count_vec,
  const vector<size_t>& index_vec)
  : ProxSeparableSum<T>(index, count, 2, interleaved, false),
  host_pt_x_(pt_x), host_pt_y_(pt_y), host_alpha_(alpha), host_beta_(beta),
  host_count_(count_vec), host_index_(index_vec)
{
}

template<typename T>
void ProxIndEpiPolyhedral1D<T>::Initialize()
{
  if(host_pt_x_.empty() ||
     host_pt_y_.empty() ||
     host_alpha_.empty() ||
     host_beta_.empty() ||
     host_index_.empty() ||
     host_count_.empty())
    throw Exception("ProxIndEpiPolyhedral1D: empty data array!");

  if(host_index_.size() != this->count() ||
     host_count_.size() != this->count())
    throw Exception("count doesn't match size of indices/counts array!");

  // Test convexity
  for(size_t i = 0; i < this->count_; i++) {
    T slope_left = host_alpha_[i];
    for(size_t j = host_index_[i]; j < host_index_[i] + host_count_[i] - 1; j++) {
      T slope_right = (host_pt_y_[j+1]-host_pt_y_[j]) / (host_pt_x_[j+1]-host_pt_x_[j]);
      if(slope_right < slope_left) {
	throw Exception("Non-convex energy passed to ProxIndEpiPolyhedral1D");
      }
      slope_left = slope_right;
    }
    if(host_beta_[i] < slope_left) {
      throw Exception("Non-convex energy passed to ProxIndEpiPolyhedral1D");
    }
  }

  // copy and allocate data on GPU
  try {
    dev_pt_x_ = host_pt_x_;
    dev_pt_y_ = host_pt_y_;
    dev_alpha_ = host_alpha_;
    dev_beta_ = host_beta_;
    dev_count_ = host_count_;
    dev_index_ = host_index_;
  }
  catch(std::bad_alloc& e) {
    throw Exception("Out of memory.");
  }
}

template<typename T>
size_t ProxIndEpiPolyhedral1D<T>::gpu_mem_amount() const
{
  return (host_pt_x_.size() + host_pt_y_.size() + host_alpha_.size() + host_beta_.size())
    * sizeof(T) + (host_count_.size() + host_index_.size()) * sizeof(size_t);
}

template<typename T>
void ProxIndEpiPolyhedral1D<T>::EvalLocal(
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

  ProxIndEpiPolyhedral1DKernel<T>
    <<<grid, block>>>(thrust::raw_pointer_cast(&(*result_beg)),
		      thrust::raw_pointer_cast(&(*arg_beg)),
		      thrust::raw_pointer_cast(dev_alpha_.data()),
		      thrust::raw_pointer_cast(dev_beta_.data()),
		      thrust::raw_pointer_cast(dev_pt_x_.data()),
		      thrust::raw_pointer_cast(dev_pt_y_.data()),
		      thrust::raw_pointer_cast(dev_count_.data()),
		      thrust::raw_pointer_cast(dev_index_.data()),
		      this->count_,
		      this->interleaved_);
}

template class ProxIndEpiPolyhedral1D<float>;
template class ProxIndEpiPolyhedral1D<double>;

} // namespace prost
