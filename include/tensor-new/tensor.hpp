#ifndef _TENSOR_NEW__TENSOR_HPP
#define _TENSOR_NEW__TENSOR_HPP

#include <memory>

#include "core/env.hpp"
#include "core/typedefs.hpp"
#include "core/utils.hpp"

namespace qtnh {
  /// General virtual tensor class
  class Tensor {
    protected:
      struct Broadcaster;

    public:
      Tensor() = delete;
      Tensor(const Tensor&) = delete;
      ~Tensor() = default;

      // This can be made constexpr in C++ 20
      virtual TT type() const noexcept { return TT::tensor; }
      
      qtnh::tidx_tup locDims() const noexcept { return loc_dims_; }
      qtnh::tidx_tup disDims() const noexcept { return dis_dims_; }
      const Broadcaster& dist() const noexcept { return bc_; }
      
      /// @brief Helper to access complete tensor dimensions. 
      /// @return Concatenated distributed and local dimensions. 
      qtnh::tidx_tup totDims() const { return utils::concat_dims(dis_dims_, loc_dims_); }
      
      /// @brief Helper to calculate size of the local part of the tensor. 
      /// @return Number of local elements in the tensor. 
      std::size_t locSize() const { return utils::dims_to_size(loc_dims_); }
      /// @brief Helper to calculate size of the distributed part of the tensor. 
      /// @return Number of ranks to which one instance of the tensor is distributed. 
      std::size_t disSize() const { return utils::dims_to_size(dis_dims_); }
      /// @brief Helper to calculate size of the entire tensor. 
      /// @return Number of elements in the entire tensor. 
      std::size_t totSize() const { return utils::dims_to_size(totDims()); }

      /// @brief Rank-unsafe method to get element and given local indices. 
      /// @param idxs Tensor index tuple indicating local position of the element. 
      /// @return Value of the element at given indices. Throws error if value is not present. 
      ///
      /// This method requires ensuring the element is present (i.e. the tensor is active)
      /// on current rank. On all active ranks, it must return an element, but different ranks  
      /// might have different values. 
      virtual qtnh::tel operator[](qtnh::tidx_tup loc_idxs) const = 0;
      /// @brief Fetch element at global indices and broadcast it to every rank. 
      /// @param idxs Tensor index tuple indicating global position of the element. 
      /// @return Value of the element at given indices. 
      ///
      /// This method doesn't require checking if the value is present or if the tensor is active. 
      /// Because of the broadcast, it is inefficient to use it too often. 
      virtual qtnh::tel fetch(qtnh::tidx_tup tot_idxs) const;


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
      /// @param str Number of times each local tensor chunk is repeated across contiguous processes. 
      /// @param cyc Number of times the entire tensor structure is repeated. 
      /// @param off Number of empty processes before the tensor begins. 
      /// @return Unique pointer to redistributed tensor, which might be of a different derived type. 
      static std::unique_ptr<Tensor> rebcast(std::unique_ptr<Tensor> tu, BcParams params) {
        return std::unique_ptr<Tensor>(tu->rebcast(params));
      }
      /// @brief Move local indices to distributed pile and distributed indices to local pile. 
      /// @param tu Unique pointer to the tensor. 
      /// @param idx_i Locations of the index to move. 
      /// @return Unique pointer to re-piled tensor, which might be of a different derived type. 
      static std::unique_ptr<Tensor> rescatter(std::unique_ptr<Tensor> tu, qtnh::tidx_tup_st idx_i) {
        return std::unique_ptr<Tensor>(tu->rescatter(idx_i));
      }


    protected:
      /// @brief Construct empty tensor of zero size within environment and with default distribution parameters. 
      /// @param env Environment to use for construction. 
      Tensor(const QTNHEnv& env);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with default distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims);
      /// @brief Construct empty tensor with given local and distributed dimensions within environment with given distribution parameters. 
      /// @param env Environment to use for construction. 
      /// @param loc_dims Local index dimensions. 
      /// @param dis_dims Distributed index dimensions. 
      /// @param params Distribution parameters of the tensor (str, cyc, off)
      Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, BcParams params);

      qtnh::tidx_tup loc_dims_;  ///< Local index dimensions. 
      qtnh::tidx_tup dis_dims_;  ///< Distributed index dimensions. 

      /// @brief Tensor broadcaster class responsible for handling how tensor is shared in distributed memory. 
      struct Broadcaster {
        qtnh::uint str;   ///< Number of times each local tensor chunk is repeated across contiguous processes. 
        qtnh::uint cyc;   ///< Number of times the entire tensor structure is repeated. 
        qtnh::uint off;   ///< Number of empty processes before the tensor begins. 
        qtnh::uint base;  ///< Base distributed size of the tensor. 

        const QTNHEnv& env;   ///< Environment to use MPI/OpenMP in. 
        MPI_Comm group_comm;  ///< Communicator that contains exactly one copy of the tensor. 
        int group_id;         ///< Rank ID within current group. 
        bool active;          ///< Flag whether the tensor is stored on calling MPI rank. 

        Broadcaster() = delete;
        Broadcaster(const QTNHEnv& env, qtnh::uint base, BcParams params);
        ~Broadcaster();

        Broadcaster& operator=(Broadcaster&& b) noexcept;

        /// @brief Helper to calculate span of the entire tensor across contiguous ranks. 
        /// @return Number of contiguous ranks that store the tensor. 
        qtnh::uint span() const noexcept { return str * base * cyc; }
        /// @brief Helper to calculate between which ranks the tensor is contained. 
        /// @return A tuple containing first and last rank that store the tensor. 
        std::pair<qtnh::uint, qtnh::uint> range() const noexcept { return { off, off + span() }; }
      };

      Broadcaster bc_;  ///< Tensor distributor. 

      // ! The following methods only work if DenseTensor overrides all of them
      /// @brief Swap indices on current tensor. 
      /// @param idx1 First index to swap. 
      /// @param idx2 Second index to swap. 
      /// @return Pointer to swapped tensor, which might be of a different derived type. 
      virtual Tensor* swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) = 0;
      /// @brief Re-broadcast current tensor. 
      /// @param params Broadcast parameters of the tensor (str, cyc, off)
      /// @return Pointer to re-broadcasted tensor, which might be of a different derived type. 
      virtual Tensor* rebcast(BcParams params) = 0;
      /// @brief Shift the border between shared and distributed dimensions by a given offset. 
      /// @param offset New offset between distributed and local dimensions – negative gathers, while positive scatters. 
      /// @return Pointer to re-scattered tensor, which might be of a different derived type. 
      virtual Tensor* rescatter(int offset) = 0;
  };
}

#endif