#ifndef _TENSOR__BASE_HPP
#define _TENSOR__BASE_HPP

#include <memory>
#include <optional>

#include "../core/env.hpp"
#include "../core/typedefs.hpp"

namespace qtnh {
  class ConvertTensor;
  class SDenseTensor;
  class DDenseTensor;

  /// Base class for tensors. 
  class Tensor {
    friend class ConvertTensor;
    friend class SDenseTensor;
    friend class DDenseTensor;

    protected:
      const QTNHEnv& env;  ///< Environment to use MPI/OpenMP in. 
      bool active;         ///< Flag whether the tensor is valid on calling MPI rank. 

      qtnh::tidx_tup dims;       ///< Global index dimensions. 
      qtnh::tidx_tup loc_dims;   ///< Local index dimensions. 
      qtnh::tidx_tup dist_dims;  ///< Distributed index dimensions.

      /// @brief Contraction dispatch to derived tensor class. 
      /// @param t Pointer to other contracted tensor. 
      /// @param ws Vector of wires for contraction. 
      /// @return Resultant tensor pointer after dispatch and contraction completes. 
      virtual Tensor* contract_disp(Tensor* t, const std::vector<qtnh::wire>& ws);

      /// @brief Contraction of derived tensor and tensor of type %ConvertTensor. 
      /// @param t Pointer to other contracted tensor of type %ConvertTensor. 
      /// @param ws Vector of wires for contraction. 
      /// @return Resultant tensor pointer after contraction completes. 
      virtual Tensor* contract(ConvertTensor* t, const std::vector<qtnh::wire>& ws);

      /// @brief Contraction of derived tensor and tensor of type %SDenseTensor. 
      /// @param t Pointer to other contracted tensor of type %SDenseTensor. 
      /// @param ws Vector of wires for contraction. 
      /// @return Resultant tensor pointer after contraction completes. 
      virtual Tensor* contract(SDenseTensor* t, const std::vector<qtnh::wire>& ws);

      /// @brief Contraction of derived tensor and tensor of type %DDenseTensor. 
      /// @param t Pointer to other contracted tensor of type %DDenseTensor. 
      /// @param ws Vector of wires for contraction. 
      /// @return Resultant tensor pointer after contraction completes. 
      virtual Tensor* contract(DDenseTensor* t, const std::vector<qtnh::wire>& ws);

    public:
      /// Empty constructor is invalid due to undefined environment. 
      Tensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      Tensor(const Tensor&) = delete;

      /// @brief Construct empty tensor of zero size within environment. 
      /// @param env Environment to use for construction. 
      Tensor(const QTNHEnv& env);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dist_dims Distributed index dimensions. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dist_dims);

      /// Default destructor. 
      virtual ~Tensor() = default;

      bool isActive() const { return active; }                         ///< Active flag getter. 
      const qtnh::tidx_tup& getDims() const { return dims; }           ///< Index dimensions getter. 
      const qtnh::tidx_tup& getLocDims() const { return loc_dims; }    ///< Local index dimensions getter. 
      const qtnh::tidx_tup& getDistDims() const { return dist_dims; }  ///< Distributed index dimensions getter. 

      std::size_t getSize() const;      ///< Get combined size of all index dimensions. 
      std::size_t getLocSize() const;   ///< Get combined size of local index dimensions. 
      std::size_t getDistSize() const;  ///< Get combined size of distributed index dimensions. 

      /// @brief Rank-safe method to get element and given global indices. 
      /// @param idxs Tensor index tuple indicating global position of the element. 
      /// @return Value of the element at given indices, or {} if value is not present. 
      ///
      /// This method doesn't require checking if value is present or if tensor is active. 
      /// It will return empty element {} if given indices are not present on current rank. 
      /// It is also guaranteed to return the same value everywhere it is present. 
      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup& idxs) const = 0;
      /// @brief Rank-safe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices, or {} if value is not present. 
      ///
      /// This method doesn't require checking if tensor is active. If that is not the case, 
      /// it will return the empty element {}. On all active ranks, it must return an element, 
      /// but different ranks might have different values. 
      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup& idxs) const = 0;
      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. As a consequence, it doesn't require wrapping the result in the optional
      /// data type. On all active ranks, it must return an element, but different ranks might 
      /// have different values. 
      virtual qtnh::tel operator[](const qtnh::tidx_tup& idxs) const = 0;

      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// 
      /// Must be overriden for every derived tensor class, as swaps depend on implementation. 
      /// For some classes it might not be a valid method. 
      virtual void swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) = 0;

      /// @brief Contract two tensors via given wires. 
      /// @param t1 Pointer to first tensor to contract. 
      /// @param t2 Pointer to second tensor to contract. 
      /// @param ws A vector of wires which indicate which pairs of indices to sum over. 
      /// @return Contracted tensor pointer. 
      static std::unique_ptr<Tensor> contract(std::unique_ptr<Tensor> t1u, std::unique_ptr<Tensor> t2u, const std::vector<qtnh::wire>& ws);
  };

  namespace ops {
    /// Print tensor elements via std::cout. 
    std::ostream& operator<<(std::ostream&, const Tensor&);
  }

  /// Virtual tensor class for when local elements are shared across all ranks
  /// (i.e. distributed dimensions are empty). 
  class SharedTensor : public virtual Tensor {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      SharedTensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      SharedTensor(const SharedTensor&) = delete;

      /// @brief Shared tensor constructor. 
      /// @param loc_dims Local index dimensions. 
      ///
      /// Create shared tensor with given local dimensions and empty distributed dimensions. 
      /// Since %Tensor is inheritted from virtally, the environment needs to be set via
      /// separate call to its constructor. 
      SharedTensor(qtnh::tidx_tup loc_dims);
      
      /// Default destructor. 
      virtual ~SharedTensor() = default;

      virtual std::optional<qtnh::tel> getEl(const qtnh::tidx_tup&) const override;
      virtual std::optional<qtnh::tel> getLocEl(const qtnh::tidx_tup&) const override;
  };

  /// Virtual tensor class that allows writing into tensor elements. 
  class WritableTensor : public virtual Tensor {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      WritableTensor() = default;
      /// Copy constructor is invalid due to potential large tensor size. 
      WritableTensor(const SharedTensor&) = delete;
      /// Default destructor. 
      virtual ~WritableTensor() = default;

      using Tensor::operator[];

      /// @brief Upade element on given global indices. 
      /// @param idxs Tensor index tuple indicating global position to be updated. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update is only executed on ranks that contain the global position. 
      /// Other ranks are unaffacted by the write. 
      virtual void setEl(const qtnh::tidx_tup& idxs, qtnh::tel el) = 0;
      /// @brief Upade element on given local indices. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// The index update is executed on all active ranks, and different values might be
      /// passed to the method on different ranks. 
      virtual void setLocEl(const qtnh::tidx_tup& idxs, qtnh::tel el) = 0;
      /// @brief Upade element on given local indices using the brackets operator. 
      /// @param idxs Tensor index tuple indicating local position to be updated. 
      /// @param el Complex number to be written at the given position. 
      ///
      /// This method is identical to setLocEl, but is included for completion, to extend 
      /// the usefulness of the square brackets operator. 
      virtual qtnh::tel& operator[](const qtnh::tidx_tup& idxs) = 0;
  };
}

#endif