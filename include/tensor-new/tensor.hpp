#ifndef _TENSOR_NEW__TENSOR_HPP
#define _TENSOR_NEW__TENSOR_HPP

#include <memory>

#include "core/env.hpp"
#include "core/typedefs.hpp"
#include "core/utils.hpp"

namespace qtnh {
  class Tensor {
    protected:
      struct Distributor;

    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      ~Tensor() = default;
      
      const qtnh::tidx_tup& locDims() const noexcept { return loc_dims_; }
      const qtnh::tidx_tup& disDims() const noexcept { return dis_dims_; }
      const Distributor& dist() const noexcept { return dist_; }
      
      /// @brief Helper to access complete tensor dimensions. 
      /// @return Concatenated distributed and local dimensions. 
      const qtnh::tidx_tup& totDims() const noexcept { return utils::concat_dims(dis_dims_, loc_dims_); }
      
      /// @brief Helper to calculate size of the local part of the tensor. 
      /// @return Number of local eements in the tensor. 
      std::size_t locSize() const noexcept { return utils::dims_to_size(loc_dims_); }
      /// @brief Helper to calculate size of the distributed part of the tensor. 
      /// @return Number of ranks to which one instance of the tensor is distributed. 
      std::size_t disSize() const noexcept { return utils::dims_to_size(dis_dims_); }
      /// @brief Helper to calculate size of the entire tensor. 
      /// @return Number of elements in the entire tensor. 
      std::size_t totSize() const noexcept { return utils::dims_to_size(totDims()); }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](const qtnh::tidx_tup& loc_idxs) const = 0;
      /// @brief Fetch element at global indices and broadcast it to every rank. 
      /// @param idxs Tensor index tuple indicating global position of the element. 
      /// @return Value of the element at given indices. 
      ///
      /// This method doesn't require checking if the value is present or if the tensor is active. 
      /// Because of the broadcast, it is inefficient to use it too often. 
      virtual qtnh::tel fetch(const qtnh::tidx_tup& tot_idxs) const;


      /// @brief Contract two tensors via given wires. 
      /// @param t1u Unique pointer to first tensor to contract. 
      /// @param t2u Unique pointer to second tensor to contract. 
      /// @param ws A vector of wires which indicate which pairs of indices to sum over. 
      /// @return Contracted tensor unique pointer. 
      static std::unique_ptr<Tensor> contract(std::unique_ptr<Tensor> t1u, std::unique_ptr<Tensor> t2u, const std::vector<qtnh::wire>& ws);

      /// @brief Swap indices on current tensor. 
      /// @param tu Unique pointer to the tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Unique to swapped tensor, which might be of a different derived type. 
      static std::unique_ptr<Tensor> swap(std::unique_ptr<Tensor> tu, qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
        return std::unique_ptr<Tensor>(tu->swap(idx1, idx2));
      }
      /// @brief Redistribute tensor to a different distribution pattern. 
      /// @param tu Unique pointer to the tensor. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      /// @return Unique pointer to redistributed tensor, which might be of a different derived type. 
      static std::unique_ptr<Tensor> redistribute(std::unique_ptr<Tensor> tu, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset) {
        return std::unique_ptr<Tensor>(tu->redistribute(stretch, cycles, offset));
      }
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param tu Unique pointer to the tensor. 
      /// @param idx_locs Locations of indices to move. 
      /// @return Unique pointer to repiled tensor, which might be of a different derived type. 
      static std::unique_ptr<Tensor> repile(std::unique_ptr<Tensor> tu, std::vector<qtnh::tidx_tup_st> idx_locs) {
        return std::unique_ptr<Tensor>(tu->repile(idx_locs));
      }


    protected:
      /// @brief Construct empty tensor of zero size within environment and with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      Tensor(const QTNHEnv& env);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution paremeters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset);

      qtnh::tidx_tup loc_dims_;  ///< Local index dimensions. 
      qtnh::tidx_tup dis_dims_;  ///< Distributed index dimensions. 

      /// @brief Tensor distributor class responsible for handling how tensor is stored in distributed memory. 
      struct Distributor {
        qtnh::uint stretch;  ///< Number of times each local tensor chunk is repeated across contiguous processes. 
        qtnh::uint cycles;   ///< Number of times the entire tensor structure is repeated. 
        qtnh::uint offset;   ///< Number of empty processes before the tensor begins. 
        qtnh::uint base;     ///< Base distributed size of the tensor. 

        const QTNHEnv& env;   ///< Environment to use MPI/OpenMP in. 
        MPI_Comm group_comm;  ///< Communicator that contains exactly one copy of the tensor. 
        bool active;          ///< Flag whether the tensor is stored on calling MPI rank. 

        Distributor(const QTNHEnv& env, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset, qtnh::uint base);

        /// @brief Helper to calculate span of the entire tensor across contiguous ranks. 
        /// @return Number of contiguous ranks that store the tensor. 
        qtnh::uint span() const noexcept { return stretch * base * cycles; }
        /// @brief Helper to calculate between which ranks the tensor is contained. 
        /// @return A tuple containing first and last rank that store the tensor. 
        std::pair<qtnh::uint, qtnh::uint> range() const noexcept { return { offset, offset + span() }; }
      };

      Distributor dist_;  ///< Tensor distributor. 

      // ! The following methods only work if DenseTensor overrides all of them
      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) = 0;
      /// @brief Redistribute current tensor. 
      /// @param stretch Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cycles Number of times the entire tensor structure is repeated. 
      /// @param offset Number of empty processes before the tensor begins. 
      /// @return Pointer to redistributed tensor, which might be of a different derived type. 
      virtual Tensor* redistribute(qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset) = 0;
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param idx_locs Locations of indices to move. 
      /// @return Pointer to repiled tensor, which might be of a different derived type. 
      virtual Tensor* repile(std::vector<qtnh::tidx_tup_st> idx_locs) = 0;
  };
}

#endif