#ifndef _TENSOR__BASE2_HPP
#define _TENSOR__BASE2_HPP

#include <memory>
#include <optional>

#include "../core/env.hpp"
#include "../core/typedefs.hpp"
#include "../core/utils.hpp"

namespace qtnh {
  class Tensor {
    protected:
      struct Distributor;

    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      ~Tensor() = default;
      
      const qtnh::tidx_tup& getLocDims() const noexcept { return loc_dims; }
      const qtnh::tidx_tup& getDisDims() const noexcept { return dis_dims; }
      const Distributor& getDistributor() const noexcept { return distributor; }
      
      /// @brief Helper getter to access complete tensor dimensions. 
      /// @return Concatenated distributed and local dimensions. 
      const qtnh::tidx_tup& getTotDims() const noexcept { return utils::concat_dims(dis_dims, loc_dims); }
      
      /// @brief Helper to calculate size of the local part of the tensor. 
      /// @return Number of local eements in the tensor. 
      std::size_t locSize() const noexcept { return utils::dims_to_size(loc_dims); }
      /// @brief Helper to calculate size of the distributed part of the tensor. 
      /// @return Number of ranks to which one instance of the tensor is distributed. 
      std::size_t disSize() const noexcept { return utils::dims_to_size(dis_dims); }
      /// @brief Helper to calculate size of the entire tensor. 
      /// @return Number of elements in the entire tensor. 
      std::size_t totSize() const noexcept { return utils::dims_to_size(getTotDims()); }

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
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
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
      /// @brief Construct empty tensor of zero size within environment and with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      Tensor(const QTNHEnv& env);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dist_dims Distributed index dimensions. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dist_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dist_dims Distributed index dimensions. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dist_dims, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset);

      qtnh::tidx_tup loc_dims;  ///< Local index dimensions. 
      qtnh::tidx_tup dis_dims;  ///< Distributed index dimensions. 

      /// @brief Tensor distributor class responsible for handling how tensor is stored in distributed memory. 
      struct Distributor {
        qtnh::uint stretch;  ///< Number of times each local tensor chunk is repeated across contiguous processes. 
        qtnh::uint cycles;   ///< Number of times the entire tensor structure is repeated. 
        qtnh::uint offset;   ///< Number of empty processes before the tensor begins. 
        qtnh::uint base;     ///< Base distributed size of the tensor. 

        const QTNHEnv& env;   ///< Environment to use MPI/OpenMP in. 
        MPI_Comm group_comm;  ///< Communicator that contains exactly one copy of the tensor. 
        bool active;          ///< Flag whether the tensor is stored on calling MPI rank. 

        /// @brief Helper to calculate span of the entire tensor across contiguous ranks. 
        /// @return Number of contiguous ranks that store the tensor. 
        qtnh::uint span() const noexcept { return stretch * base * cycles; }
        /// @brief Helper to calculate between which ranks the tensor is contained. 
        /// @return A tuple containing first and last rank that store the tensor. 
        std::pair<qtnh::uint, qtnh::uint> range() const noexcept { return { offset, offset + span() }; }
      };

      Distributor distributor;  ///< Tensor distributor. 
  };
}

#endif