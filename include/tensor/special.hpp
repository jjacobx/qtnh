#ifndef _TENSOR__SPECIAL_HPP
#define _TENSOR__SPECIAL_HPP

#include "core/typedefs.hpp"
#include "tensor/base.hpp"
#include "tensor/dense.hpp"

namespace qtnh {
  /// Swap tensor class, used to allow encoding index swaps as swaps in the tensor network. 
  /// Two indices of another tensor can be contracted with a swap tensor, and the result
  /// of the contraction is exchange of said indices. If swaps are not allowed for a given 
  /// tensor, contracting it with a swap tensor will throw an error. 
  class SwapTensor : public SharedTensor {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      SwapTensor() = delete;
      /// Copy constructor is invalid by convention. 
      SwapTensor(const SwapTensor&) = delete;

      /// @brief Construct a swap tensor with two given input dimensions. 
      /// @param env Environment to use for construction. 
      /// @param n1 Dimensions of the first swapped index. 
      /// @param n2 Dimensions of the second swapped index. 
      ///
      /// The created tensor has 4 indices of dimensions (n1, n2, n1, n2). 
      SwapTensor(const QTNHEnv& env, std::size_t n1, std::size_t n2);

      /// Default destructor. 
      ~SwapTensor() = default;

      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;
      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;

    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
  };

  /// The identity tensor has no effect on the tensor it is contracted with, but can be useful
  /// for simplifying the tensor network. 
  class IdentityTensor : public SharedTensor {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      IdentityTensor() = delete;
      /// Copy constructor is invalid by convention. 
      IdentityTensor(const IdentityTensor&) = delete;
      
      /// @brief Create identity tensor with given input dimensions. 
      /// @param env Environment to use for construction. 
      /// @param in_dims Input index dimensions. 
      ///
      /// Constructs a tensor with the same output dimensions as input dimensions. When 
      /// contructing wiht input dimensions, output index elements will remain the same. 
      IdentityTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims);

      /// Default destructor. 
      ~IdentityTensor() = default;

      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;
      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;

    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
  };

  /// The convert tensor acts as an identity tensor, but it also modifies the type of indices. 
  /// It converts local indices to distributed and vice versa. As a result, it can only be applied 
  /// to first local or last distributed indices. 
  class ConvertTensor : public IdentityTensor {
    friend class SDenseTensor;
    friend class DDenseTensor;

    public:
      /// Empty constructor is invalid due to undefined environment. 
      ConvertTensor() = delete;
      /// Copy constructor is invalid by convention. 
      ConvertTensor(const ConvertTensor&) = delete;

      /// @brief Create convert tensor with given input dimensions. 
      /// @param env Environment to use for construction. 
      /// @param in_dims Input index dimensions. 
      ///
      /// Constructs a tensor with the same output dimensions as input dimensions. Under 
      /// contraction, the tensor elements are repeated, but the type of input indices is 
      /// changed from local to distributed and vice versa.  
      ConvertTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims);

      /// Default destructor. 
      ~ConvertTensor() = default;
    
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
  };
}

#endif