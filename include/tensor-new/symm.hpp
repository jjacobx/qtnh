#ifndef _TENSOR_NEW__SYMM_HPP
#define _TENSOR_NEW__SYMM_HPP

#include "tensor-new/dense.hpp"

namespace qtnh {
  class SymmTensorBase;
  class SymmTensor;
  class SwapTensor;

  class SymmTensorBase : public DenseTensorBase {
    public: 
      SymmTensorBase() = delete;
      SymmTensorBase(const SymmTensorBase&) = delete;
      ~SymmTensorBase() = default;

      virtual TT type() const noexcept override { return TT::symmTensorBase; }

      qtnh::tidx_tup disInDims() const {
        return qtnh::tidx_tup(dis_dims_.begin(), dis_dims_.begin() + n_dis_in_dims_);
      }
      qtnh::tidx_tup locInDims() const {
        return qtnh::tidx_tup(loc_dims_.begin(), loc_dims_.begin() + (loc_dims_.size() / 2) - n_dis_in_dims_);
      }
      qtnh::tidx_tup totInDims() const {
        return utils::concat_dims(disInDims(), locInDims());
      }

      qtnh::tidx_tup disOutDims() const {
        return qtnh::tidx_tup(dis_dims_.begin() + n_dis_in_dims_, dis_dims_.end());
      }
      qtnh::tidx_tup locOutDims() const {
        return qtnh::tidx_tup(loc_dims_.begin() + (loc_dims_.size() / 2) - n_dis_in_dims_, loc_dims_.end());
      }
      qtnh::tidx_tup totOutDims() const {
        return utils::concat_dims(disOutDims(), locOutDims());
      }

      static std::unique_ptr<SymmTensor> toSymm(std::unique_ptr<SymmTensorBase> tu) {
        return std::unique_ptr<SymmTensor>(tu->toSymm());
      }

    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, DistParams params);

      /// @brief Convert any derived tensor to symmetric tensor
      /// @return Symmetric tensor equivalent to calling tensor
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
      /// @return Pointer to repiled tensor, which might be of a different derived type. 
      virtual Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;

      qtnh::tidx_tup_st n_dis_in_dims_;
  };

  class SymmTensor : public SymmTensorBase {
    public:
      SymmTensor() = delete;
      SymmTensor(const SymmTensor&) = delete;
      ~SymmTensor() = default;

      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param els Complex vector of local elements. 
      SymmTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel> els);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
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
      virtual qtnh::tel operator[](const qtnh::tidx_tup& loc_idxs) const override;
      /// @brief Set element on given local indices. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      ///
      /// The index update is executed on all active ranks, and different values might be
      /// passed to the method on different ranks. 
      qtnh::tel& operator[](const qtnh::tidx_tup& loc_idxs);
      /// @brief Set element on given global indices. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update will do nothing on ranks that do not contain the element on given indices. 
      void put(const qtnh::tidx_tup& loc_idxs, qtnh::tel el);

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
      /// @return Pointer to repiled tensor, which might be of a different derived type. 
      virtual Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;

    private:
      std::vector<qtnh::tel> loc_els;
  };
}

#endif