#ifndef _TENSOR_NEW__SYMM_HPP
#define _TENSOR_NEW__SYMM_HPP

#include "tensor-new/dense.hpp"

namespace qtnh {
  class SymmTensorBase;
  class SymmTensor;
  class SwapTensor;

  /// Symmetric tensor base virtual class, which restricts dense tensors by assuming dimensions can be split into equal input and output. 
  /// While in/out local and distributed dimension may differ, the concatenated in/out parts have to be exactly the same. 
  class SymmTensorBase : public DenseTensorBase {
    public: 
      SymmTensorBase() = delete;
      SymmTensorBase(const SymmTensorBase&) = delete;
      ~SymmTensorBase() = default;

      virtual TT type() const noexcept override { return TT::symmTensorBase; }

      /// Get distributed input dimensions. 
      qtnh::tidx_tup disInDims() const {
        return qtnh::tidx_tup(dis_dims_.begin(), dis_dims_.begin() + n_dis_in_dims_);
      }
      /// Get local input dimensions. 
      qtnh::tidx_tup locInDims() const {
        return qtnh::tidx_tup(loc_dims_.begin(), loc_dims_.begin() + (loc_dims_.size() / 2) - n_dis_in_dims_);
      }
      /// Get input dimensions, which must be equal to output dimensions. 
      qtnh::tidx_tup totInDims() const {
        return utils::concat_dims(disInDims(), locInDims());
      }

      /// Get distributed output dimensions. 
      qtnh::tidx_tup disOutDims() const {
        return qtnh::tidx_tup(dis_dims_.begin() + n_dis_in_dims_, dis_dims_.end());
      }
      /// Get local output dimensions. 
      qtnh::tidx_tup locOutDims() const {
        return qtnh::tidx_tup(loc_dims_.begin() + (loc_dims_.size() / 2) - n_dis_in_dims_, loc_dims_.end());
      }
      /// Get output dimensions, which must be equal to input dimensions. 
      qtnh::tidx_tup totOutDims() const {
        return utils::concat_dims(disOutDims(), locOutDims());
      }

      /// @brief Convert any derived tensor to writable symmetric tensor
      /// @param tu Unique pointer to derived symmetric tensor to convert. 
      /// @return Unique pointer to an equivalent writable symmetric tensor. 
      static std::unique_ptr<SymmTensor> toSymm(std::unique_ptr<SymmTensorBase> tu) {
        return std::unique_ptr<SymmTensor>(tu->toSymm());
      }

    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, DistParams params);

      /// @brief Convert any derived tensor to writable symmetric tensor
      /// @return Pointer to an equivalent writable symmetric tensor. 
      virtual SymmTensor* toSymm();

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Redistribute current tensor. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      virtual Tensor* redistribute(DistParams params) override;
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param idx_locs Locations of indices to move. 
      /// @return Pointer to re-piled tensor, which might be of a different derived type. 
      virtual Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;

      qtnh::tidx_tup_st n_dis_in_dims_;  ///< Number of distributed input dimensions. 
  };

  /// Writable general symmetric tensor class, which allows direct access to all elements. Restrictions for symmetric tensors apply. 
  class SymmTensor : public SymmTensorBase {
    public:
      SymmTensor() = delete;
      SymmTensor(const SymmTensor&) = delete;
      ~SymmTensor() = default;

      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param els Complex vector of local elements. 
      SymmTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel> els);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param els Complex vector of local elements. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      SymmTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel> els, DistParams params);

      virtual TT type() const noexcept override { return TT::symmTensor; }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](qtnh::tidx_tup loc_idxs) const override;
      /// @brief Set element on given local indices. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      ///
      /// The index update is executed on all active ranks, and different values might be
      /// passed to the method on different ranks. 
      qtnh::tel& operator[](qtnh::tidx_tup loc_idxs);
      /// @brief Set element on given global indices. 
      /// @param tot_idxs Tensor index tuple indicating global position to be updated. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update will do nothing on ranks that do not contain the element on given indices. 
      void put(qtnh::tidx_tup tot_idxs, qtnh::tel el);

    protected:
      /// @brief Convert any derived tensor to symmetric tensor
      /// @return Symmetric tensor equivalent to calling tensor
      virtual SymmTensor* toSymm() override { return this; }

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Redistribute current tensor. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      virtual Tensor* redistribute(DistParams params) override;
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param idx_locs Locations of indices to move. 
      /// @return Pointer to re-piled tensor, which might be of a different derived type. 
      virtual Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;

    private:
      std::vector<qtnh::tel> loc_els;  ///< Local elements. 
  };

  /// Rank 4 symmetric swap tensor for swapping two indices with dimension n. The swap tensor must have dimensions (n, n, n, n). 
  class SwapTensor : public SymmTensorBase {
    public:
      SwapTensor() = delete;
      SwapTensor(const SymmTensor&) = delete;
      ~SwapTensor() = default;

      /// @brief Construct rank 4 swap tensor with given single index size within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of swapped indices (both are expected to have the same size)
      SwapTensor(const QTNHEnv& env, std::size_t n);
      /// @brief Construct rank 4 swap tensor with given index size within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param n Dimension of swapped indices (both are expected to have the same size)
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      SwapTensor(const QTNHEnv& env, std::size_t n, DistParams params);

      virtual TT type() const noexcept override { return TT::swapTensor; }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](qtnh::tidx_tup loc_idxs) const override;

    protected:
      /// @brief Redistribute current tensor. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      virtual Tensor* redistribute(DistParams params) override;
  };
}

#endif