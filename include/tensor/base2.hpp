#ifndef _TENSOR__BASE2_HPP
#define _TENSOR__BASE2_HPP

#include <memory>
#include <optional>

#include "../core/env.hpp"
#include "../core/typedefs.hpp"

namespace qtnh {
  class Tensor {
    public:
      /// Empty constructor is invalid due to undefined environment. 
      Tensor() = delete;
      /// Copy constructor is invalid due to potential large tensor size. 
      Tensor(const Tensor&) = delete;
      /// Default destructor. 
      ~Tensor() = default;

      struct Distributor {
        qtnh::uint stretch;
        qtnh::uint cycles;
        qtnh::uint offset;
        qtnh::uint base;

        const QTNHEnv& env;

        MPI_Comm group_comm;
        bool active;

        qtnh::uint getSpan() { return stretch * base * cycles; }
        qtnh::uint getLowerRange() { return offset; }
        qtnh::uint getUpperRange() { return offset + getSpan(); }
      };

      const qtnh::tidx_tup& getDims() const { return dims; }           ///< Index dimensions getter. 
      const qtnh::tidx_tup& getLocDims() const { return loc_dims; }    ///< Local index dimensions getter. 
      const qtnh::tidx_tup& getDistDims() const { return dist_dims; }  ///< Distributed index dimensions getter. 

      const Distributor& getDistributor() const { return distributor; }  ///< Tensor distributor getter. 

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
      /// @brief Redistribute current tensor. 
      /// @param stretch How many times each local element is repeated. 
      /// @param cycles How many times the entire tensor structure is repeated. 
      /// @param offset Number of emoty processes before the tensor begins. 
      /// 
      /// Must be overriden for every derived tensor class, as distribution depends on implementation. 
      virtual void redistribute(qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset) = 0;

      /// @brief Contract two tensors via given wires. 
      /// @param t1u Unique pointer to first tensor to contract. 
      /// @param t2u Unique pointer to second tensor to contract. 
      /// @param ws A vector of wires which indicate which pairs of indices to sum over. 
      /// @return Contracted tensor unique pointer. 
      static std::unique_ptr<Tensor> contract(std::unique_ptr<Tensor> t1u, std::unique_ptr<Tensor> t2u, const std::vector<qtnh::wire>& ws);

    protected:
      /// @brief Construct empty tensor of zero size within environment. 
      /// @param env Environment to use for construction. 
      Tensor(const QTNHEnv& env);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dist_dims Distributed index dimensions. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dist_dims);

      Distributor distributor;  ///< Tensor distributor. 

      qtnh::tidx_tup dims;       ///< Global index dimensions. 
      qtnh::tidx_tup loc_dims;   ///< Local index dimensions. 
      qtnh::tidx_tup dist_dims;  ///< Distributed index dimensions. 
  };
}

#endif