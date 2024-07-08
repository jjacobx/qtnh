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
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset). 
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
      /// @return Pointer to re-piled tensor, which might be of a different derived type. 
      virtual Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;
  };

  /// Writable diagonal tensor class, which allows direct access to diagonal elements. 
  /// Only non-zero elements are stored in a vector, significantly reducing memory consumption. 
  class DiagTensor : public DiagTensorBase {
    public: 
      DiagTensor() = delete;
      DiagTensor(const DiagTensor&) = delete;
      ~DiagTensor() = default;

      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param diag_els Complex vector of local diagonal elements. 
      DiagTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel> diag_els);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param diag_els Complex vector of local diagonal elements. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset). 
      DiagTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel> diag_els, DistParams params);

      virtual TT type() const noexcept override { return TT::diagTensor; }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](const qtnh::tidx_tup& loc_idxs) const override;
      /// @brief Set element on given local indices, which must be diagonal. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      ///
      /// The index update is executed on all active ranks, and different values might be
      /// passed to the method on different ranks. Non-diagonal updates will throw an error. 
      qtnh::tel& operator[](const qtnh::tidx_tup& loc_idxs);
      /// @brief Set element on given global indices. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update will do nothing on ranks that do not contain the element on given indices, 
      /// or if the element is non-diagonal. 
      void put(const qtnh::tidx_tup& loc_idxs, qtnh::tel el);

    protected: 
      /// @brief Convert any derived tensor to writable diagonal tensor
      /// @return Pointer to an equivalent writable diagonal tensor. 
      virtual DiagTensor* toDiag() override { return this; }

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
      std::vector<qtnh::tel> loc_diag_els;  ///< Local diagonal elements. 
  };

  /// Identity diagonal tensor class. Has symmetric total dimensions and 1s on a diagonal – all other elements are zero. 
  /// Can be used for making indices local/distributed by using different input/output local-distributed dimension splits. 
  class IdenTensor : public DiagTensorBase {
    public:
      IdenTensor() = delete;
      IdenTensor(const IdenTensor&) = delete;
      ~IdenTensor() = default;

      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      IdenTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param n_dis_in_dims Number of distributed input dimensions
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset). 
      IdenTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, DistParams params);

      virtual TT type() const noexcept override { return TT::idenTensor; }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](const qtnh::tidx_tup& loc_idxs) const override;

    protected:
      /// @brief Redistribute current tensor. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      virtual Tensor* redistribute(DistParams params) override;
  };
}

#endif