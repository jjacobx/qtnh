#ifndef _TENSOR__DENSE_HPP
#define _TENSOR__DENSE_HPP

#include "core/typedefs.hpp"
#include "tensor/base.hpp"

namespace qtnh {
  /// Virtual tensor class which assumes that all elements are stored in a vector. 
  /// Local elements can be the same on all ranks (shared tensor) or different (distributed tensor). 
  class DenseTensor : public WritableTensor {
    protected:
      std::vector<qtnh::tel> loc_els; ///< Local elements. 
    
    public:
      /// Empty constructor is invalid due to undefined environment. 
      DenseTensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      DenseTensor(const DenseTensor&) = delete;

      /// @brief Dense tensor constructor. 
      /// @param els Complex vector of local elements. 
      ///
      /// The local elements might be different on different rank, but the constructor behaviour 
      /// is determined by the derived type from dense tensor. Since this is a virtual class, construction 
      /// of dimensions also needs to be handled by the derived class. 
      DenseTensor(std::vector<qtnh::tel> els);

      /// Default destructor. 
      ~DenseTensor() = default;

      virtual qtnh::tel operator[](const qtnh::tidx_tup&) const override;
      virtual qtnh::tel& operator[](const qtnh::tidx_tup&) override;
  };

  /// Shared dense tensor class which ensures that the dense tensor is active and has the same
  /// values across all ranks. 
  class SDenseTensor : public DenseTensor, public SharedTensor {
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(ConvertTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;
      
    public: 
      /// Empty constructor is invalid due to undefined environment. 
      SDenseTensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      SDenseTensor(const SDenseTensor&) = delete;

      /// @brief Create shared dense tensor with given dimensions and elements. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      ///
      /// Uses %SharedTensor constructor to initialise dimensions. Distributed dimensions are empty by convention. 
      /// The size of local elements must be equal to the size of the tensor. For safety, the elements on the root 
      /// rank are broadcast to other ranks to ensure all ranks have equal local vecotors. This can be disabled by 
      /// setting DEF_STENSOR_BCAST=0 during compilation, for faster, but less safe execution. 
      SDenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel> els);
      /// @brief Create shared dense tensor with given dimensions and elements, and with an explicit scatter flag. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param bcast Explicit flag to control root elements broadcast. 
      ///
      /// Works the same as the other constructor, but the broadcast of root elements is controlled directly, 
      /// regardless of the DEF_STENSOR_BCAST flag. 
      SDenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, std::vector<qtnh::tel> els, bool bcast);

      /// Default destructor. 
      ~SDenseTensor() = default;

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
  };

  /// Distributed dense tensor class that splits the tensor along first n indices, the total size of which
  /// determines the number of active ranks. The values of local elements of the tensor vary across active 
  /// ranks. It is forbidden to contract distributed indices – instead they have to be swapped to the local 
  /// regime, and only then contracted with another shared index. 
  class DDenseTensor : public DenseTensor {
    private:
      virtual Tensor* contract_disp(Tensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(ConvertTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(SDenseTensor*, const std::vector<qtnh::wire>&) override;
      virtual Tensor* contract(DDenseTensor*, const std::vector<qtnh::wire>&) override;

    public:
      /// Empty constructor is invalid due to undefined environment. 
      DDenseTensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      DDenseTensor(const SDenseTensor&) = delete;

      /// @brief Create distributed dense tensor with given dimensions and local elements, 
      /// and first n dimensions distributed. 
      /// @param env Environment to use for construction. 
      /// @param dims Global index dimensions. 
      /// @param els Complex vector of local elements. 
      /// @param n Number of distributed indices, taken from first elements of dims. 
      ///
      /// Local elements can (and probably should) be different on all active ranks. Only processes inside 
      /// the size of first n indices are marked as active. 
      DDenseTensor(const QTNHEnv& env, qtnh::tidx_tup dims, std::vector<qtnh::tel> els, qtnh::tidx_tup_st n);

      /// Default destructor. 
      ~DDenseTensor() = default;

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup&) const override;
      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup& idxs) const override;
      virtual void setEl(const qtnh::tidx_tup&, qtnh::tel) override;
      virtual void setLocEl(const qtnh::tidx_tup&, qtnh::tel) override;

      /// Swapping any of the distributed indices requires communication. 
      virtual void swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) override;

      /// @brief Scatter first n local indices. 
      /// @param n Number of local indices to be scattered. 
      ///
      /// Used to distribute the tensor further. The number of active ranks is multiplied by the size of first 
      /// n local indices, and the additional ranks must be avaialble. 
      void scatter(tidx_tup_st n);
      /// @brief Gather last n distributed indices. 
      /// @param n Number of distributed indices to be gathered. 
      ///
      /// Used to share more indices of the tensor. The number of active ranks is divided by the size of last 
      /// n distributed indices, while the local memory is multiplied by it, and must be available at execution. 
      void gather(tidx_tup_st n);
      /// @brief Gather all indices and share the tensor across all ranks. 
      /// @return Pointer to a shared dense tensor. 
      ///
      /// Converts from %DDenseTensor to %SDenseTensor. It needs to be ensured that enough memory is abailable 
      /// on each rank to fir t the entire tensor. 
      SDenseTensor* share();

      /// @brief Repeat the entire tensor n times along the distributed indices. 
      /// @param n Number of times to repeat the tensor. 
      void rep_all(std::size_t n);
      /// @brief Repeat each of the local parts of the tensor n times along the distributed indices. 
      /// @param n Number of times to repeat the elements. 
      void rep_each(std::size_t n);
  };
}

#endif