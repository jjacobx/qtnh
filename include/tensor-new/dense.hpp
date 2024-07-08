#ifndef _TENSOR_NEW__DENSE_HPP
#define _TENSOR_NEW__DENSE_HPP

#include "tensor-new/tensor.hpp"

namespace qtnh {
  class DenseTensorBase;
  class DenseTensor;

  /// Dense tensor base virtual class, which assumes that all local elements can be stored in a vector. 
  class DenseTensorBase : public Tensor {
    public:
      DenseTensorBase() = delete;
      DenseTensorBase(const DenseTensorBase&) = delete;
      ~DenseTensorBase() = default;

      virtual TT type() const noexcept override { return TT::denseTensorBase; }

      /// @brief Convert any derived tensor to writable dense tensor
      /// @param tu Unique pointer to derived dense tensor to convert. 
      /// @return Unique pointer to an equivalent writable dense tensor. 
      static std::unique_ptr<DenseTensor> toDense(std::unique_ptr<DenseTensorBase> tu) {
        return std::unique_ptr<DenseTensor>(tu->toDense());
      }
    
    protected:
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, DistParams params);

      /// @brief Convert any derived tensor to writable dense tensor. 
      /// @return Pointer to equivalent writable dense tensor. 
      virtual DenseTensor* toDense();

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

  /// Writable dense tensor class, which allows direct access to all elements. 
  class DenseTensor : public DenseTensorBase {
    public:
      DenseTensor() = delete;
      DenseTensor(const DenseTensor&) = delete;
      ~DenseTensor() = default;

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
      /// @param params Distribution parameters of the tensor (stretch, cycles, offset)
      DenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel> els, DistParams params);

      virtual TT type() const noexcept override { return TT::denseTensor; }

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
      /// @brief Convert any derived tensor to writable dense tensor. 
      /// @return Pointer to equivalent writable dense tensor. 
      virtual DenseTensor* toDense() noexcept override { return this; }
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
      std::vector<qtnh::tel> loc_els;  ///< Local elements. 
  };
}

#endif