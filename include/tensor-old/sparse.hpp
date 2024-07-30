#ifndef _TENSOR__SPARSE_HPP
#define _TENSOR__SPARSE_HPP

#include "core/typedefs.hpp"
#include "tensor/base.hpp"
#include "tensor/dense.hpp"

namespace qtnh {
  /// Virtual tensor class which assumes that diagonal elements are stored in a vector, and off-diagonal elements are zero. 
  /// Local elements can be the same on all ranks (shared tensor) or different (distributed tensor). 
  class SparseTensor : public WritableTensor {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      SparseTensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      SparseTensor(const SparseTensor&) = delete;

      /// @brief Sparse tensor constructor. 
      /// @param els Complex vector of local diagonal elements. 
      ///
      /// The local elements might be different on different rank, but the constructor behaviour 
      /// is determined by the derived type from sparse tensor. Since this is a virtual class, construction 
      /// of dimensions also needs to be handled by the derived class. 
      SparseTensor(std::vector<qtnh::tel> diag_els);

      /// Default destructor. 
      virtual ~SparseTensor() = default;

      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;
      virtual qtnh::tel& operator[](const qtnh::tidx_tup&) override;
    
    protected:
      std::vector<qtnh::tel> diag_els; ///< Local diagonal elements. 
  };

  /// Shared dense tensor class which ensures that the dense tensor is active and has the same
  /// values across all ranks. 
  class SSparseTensor : public SparseTensor, public SharedTensor {
    public: 
      /// Empty constructor is invalid due to undefined environment. 
      SSparseTensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      SSparseTensor(const SSparseTensor&) = delete;

      /// @brief Create shared dense tensor with given dimensions and elements. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      ///
      /// Uses %SharedTensor constructor to initialise dimensions. Distributed dimensions are empty by convention. 
      /// The size of local elements must be equal to the size of the tensor. For safety, the elements on the root 
      /// rank are broadcast to other ranks to ensure all ranks have equal local vecotors. This can be disabled by 
      /// setting DEF_STENSOR_BCAST=0 during compilation, for faster, but less safe execution. 
      SSparseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel> els);
      /// @brief Create shared dense tensor with given dimensions and elements, and with an explicit scatter flag. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param bcast Explicit flag to control root elements broadcast. 
      ///
      /// Works the same as the other constructor, but the broadcast of root elements is controlled directly, 
      /// regardless of the DEF_STENSOR_BCAST flag. 
      SSparseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel> els, bool bcast);

      /// Default destructor. 
      ~SSparseTensor() = default;

      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) override;

      /// Any indices of a shared tensor can be swapped. 
      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;

      /// @brief Distribute n first indices. 
      /// @param n Number of indices to distribute. 
      /// @return Pointer to %DDenseTensor with n distributed indices. 
      ///
      /// The result tensor is distributed to the number of ranks equivalent to the size of first n indices. 
      /// It is better to distribute all needed indices here, as this is an efficient method, and it doesn't
      /// Require communication. In contrast, distributing the %DDenseTensor further does require communication. 
      DDenseTensor* distribute(tidx_tup_st n);
    
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(ConvertTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
  };
}

#endif