#ifndef PROST_BLOCK_SPARSE_KRON_ID_HPP_
#define PROST_BLOCK_SPARSE_KRON_ID_HPP_

#include "prost/linop/block.hpp"

namespace prost {

///
/// \brief Linear operator composed of many small sparse matrices.
///        Implements linear operator for kron(M, speye(diaglength)),
///        where M is a small sparse matrix.
///
template<typename T>
class BlockSparseKronId : public Block<T>
{
  BlockSparseKronId(size_t row, size_t col, size_t nrows, size_t ncols);
  
 public:
  static BlockSparseKronId<T> *CreateFromCSC(
      size_t row,
      size_t col,
      size_t diaglength,
      int m,
      int n,
      int nnz,
      const vector<T>& val,
      const vector<int32_t>& ptr,
      const vector<int32_t>& ind);

  virtual ~BlockSparseKronId() {}

  virtual void Initialize();

  virtual T row_sum(size_t row, T alpha) const;
  virtual T col_sum(size_t col, T alpha) const;

  virtual size_t gpu_mem_amount() const;

  /// \brief Important: has to be called once during initializaiton.
  static void ResetConstMem() {
    cmem_counter_nnz_ = cmem_counter_rows_ = cmem_counter_cols_ = 0;
  }

 protected:
  virtual void EvalLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);

  virtual void EvalAdjointLocalAdd(
    const typename device_vector<T>::iterator& res_begin,
    const typename device_vector<T>::iterator& res_end,
    const typename device_vector<T>::const_iterator& rhs_begin,
    const typename device_vector<T>::const_iterator& rhs_end);

 private:
  /// \brief Size of diagonal identity matrix Id for kron(M, Id).
  size_t diaglength_;

  /// \brief Number of rows in small sparse matrix M.
  size_t mat_nrows_;

  /// \brief Number of columns in small sparse matrix M.
  size_t mat_ncols_;

  /// \brief Number of non-zero elements in small sparse matrix M.
  size_t mat_nnz_;

  /// \brief Is the sparse matrix stored in constant memory?
  bool in_cmem_;

  /// \brief Offset into constant memory.
  size_t cmem_offset_nnz_;
  size_t cmem_offset_rows_;
  size_t cmem_offset_cols_;
  
  /// \brief GPU data for small sparse matrix M (used if not stored in const-mem).
  device_vector<int32_t> ind_, ind_t_;
  device_vector<int32_t> ptr_, ptr_t_;
  device_vector<float> val_, val_t_;
  
  /// \brief Host data for small sparse matrix M.
  vector<int32_t> host_ind_, host_ind_t_;
  vector<int32_t> host_ptr_, host_ptr_t_;
  vector<float> host_val_, host_val_t_;

  /// Constant memory counter.
  static size_t cmem_counter_nnz_;
  static size_t cmem_counter_rows_;
  static size_t cmem_counter_cols_;
};

} // namespace prost

#endif // PROST_BLOCK_SPARSE_KRON_ID_HPP_
