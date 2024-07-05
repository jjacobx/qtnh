#ifndef _TENSOR_NEW__SYMM_HPP
#define _TENSOR_NEW__SYMM_HPP

#include "tensor-new/dense.hpp"

namespace qtnh {
  class SymmTensorBase;
  class SymmTensor;
  class SwapTensor;

  class SymmTensorBase : DenseTensorBase {
    public: 
      SymmTensorBase() = delete;
      SymmTensorBase(const SymmTensorBase&) = delete;
      ~SymmTensorBase() = default;

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

      virtual SymmTensor* toSymm();

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Redistribute current tensor. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      Tensor* redistribute(DistParams params) override;
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param idx_locs Locations of indices to move. 
      /// @return Pointer to repiled tensor, which might be of a different derived type. 
      Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;

      qtnh::tidx_tup_st n_dis_in_dims_;

  };
}

#endif