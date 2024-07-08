#ifndef _TENSOR_NEW__DIAG_HPP
#define _TENSOR_NEW__DIAG_HPP

#include "tensor-new/symm.hpp"

namespace qtnh {
  class DiagTensorBase;
  class DiagTensor;
  class IdenTensor;

  /// Diagonal tensor base virtual class, which assumes that a symmetric tensors only has non-zero elements on a diagonal. 
  /// The diagonal is defined by the same total input and output indices. Symmetric tensor dimension restrictions apply. 
  class DiagTensorBase : public SymmTensorBase {
    public: 
      DiagTensorBase() = delete;
      DiagTensorBase(const DiagTensorBase&) = delete;
      ~DiagTensorBase() = default;

      virtual TT type() const noexcept override { return TT::diagTensorBase; }

      /// @brief Convert any derived tensor to writable diagonal tensor
      /// @param tu Unique pointer to derived diagonal tensor to convert. 
      /// @return Unique pointer to an equivalent writable diagonal tensor. 
      static std::unique_ptr<DiagTensor> toDiag(std::unique_ptr<DiagTensorBase> tu) {
        return std::unique_ptr<DiagTensor>(tu->toDiag());
      }

    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, DistParams params);

      /// @brief Convert any derived tensor to writable diagonal tensor
      /// @return Pointer to an equivalent writable diagonal tensor. 
      virtual DiagTensor* toDiag();

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
  };

}

#endif