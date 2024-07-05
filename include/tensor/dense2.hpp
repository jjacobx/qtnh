#ifndef _TENSOR__DENSE2_HPP
#define _TENSOR__DENSE2_HPP

#include "core/typedefs.hpp"
#include "tensor/tensor.hpp"

namespace qtnh {
  class DenseTensorBase;
  class DenseTensor;

  /// Virtual tensor class which assumes that all elements are stored in a vector. 
  /// Local elements can be the same on all ranks (shared tensor) or different (distributed tensor). 
  class DenseTensorBase : public Tensor {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      DenseTensorBase() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      DenseTensorBase(const DenseTensorBase&) = delete;

      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param els Complex vector of local elements. 
      DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel> els);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel> els, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset);

      /// Default destructor. 
      virtual ~DenseTensorBase() = default;

      static std::unique_ptr<DenseTensor> toDense(std::unique_ptr<DenseTensorBase> tu) noexcept {
        return std::unique_ptr<DenseTensor>(tu->toDense());
      }
    
    protected:
      /// @brief Convert tensor to dense tensor format, which is most general. 
      /// @return Pointer to equivalent dense tensor. 
      virtual DenseTensor* toDense() noexcept;

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Redistribute current tensor. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      Tensor* redistribute(qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset) override;
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param idx_locs Locations of indices to move. 
      /// @return Pointer to repiled tensor, which might be of a different derived type. 
      Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;

      std::vector<qtnh::tel> loc_els; ///< Local elements. 
  };

    class DenseTensor : public DenseTensorBase {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      DenseTensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      DenseTensor(const DenseTensor&) = delete;

      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param els Complex vector of local elements. 
      DenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel> els);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      DenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel> els, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset);

      /// Default destructor. 
      virtual ~DenseTensor() = default;

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      qtnh::tel operator[](const qtnh::tidx_tup& loc_idxs) const override;
      /// @brief Insert element on given local indices. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update is executed on all active ranks, and different values might be
      /// passed to the method on different ranks. 
      // qtnh::tel& operator[](const qtnh::tidx_tup& loc_idxs) override;
    
    protected:
      /// @brief Convert tensor to dense tensor format, which is most general. 
      /// @return Pointer to equivalent dense tensor. 
      virtual DenseTensor* toDense() noexcept override { return this; }
      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) override;
      /// @brief Redistribute current tensor. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      Tensor* redistribute(qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset) override;
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param idx_locs Locations of indices to move. 
      /// @return Pointer to repiled tensor, which might be of a different derived type. 
      Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) override;

      std::vector<qtnh::tel> loc_els; ///< Local elements. 
  };
}

#endif