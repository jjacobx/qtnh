#ifndef _TENSOR__NETWORK_HPP
#define _TENSOR__NETWORK_HPP

#include <memory>
#include <unordered_map>

#include "core/typedefs.hpp"
#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/diag.hpp"
#include "tensor/symm.hpp"


namespace qtnh {
  /// Storage for tensors and bonds connecting them. 
  class TensorNetwork {
    public:
      TensorNetwork();
      TensorNetwork(const TensorNetwork&) = delete;
      ~TensorNetwork() = default;

      /// Used for storage of wires for tensor contraction, together with their two target tensors. 
      struct Bond {
        std::pair<qtnh::uint, qtnh::uint> tensor_ids;  ///< IDs of two target tensors. 
        std::vector<qtnh::wire> wires;                 ///< Wires connecting tensor indices. 

        /// @brief General bond constructor, can specify if contraction is in-place. 
        /// @param tids Pair of IDs of target tensors. 
        /// @param ws Vector of wires for contraction. 
        Bond(std::pair<qtnh::uint, qtnh::uint> tids, std::vector<qtnh::wire> ws);

        /// Default destructor. 
        ~Bond() = default;
      };

      /// @brief Get tensor with ID. 
      /// @param tid Tensor ID in the map. 
      /// @return Pointer to tensor with given ID. 
      Tensor* tensor(qtnh::uint tid);
      /// @brief Get bond with ID. 
      /// @param bid Bond ID in the map. 
      /// @return Copy of the bond with given ID. 
      const Bond& bond(qtnh::uint bid);
      /// @brief Get all tensor IDs in the network. 
      /// @return A vector of all tensor IDs present in the network. 
      std::vector<qtnh::uint> tensorIDs();
      /// @brief Get all bond IDs in the network. 
      /// @return A vector of all bond IDs present in the network. 
      std::vector<qtnh::uint> bondIDs();

      /// @brief Erase and extract tensor with ID. 
      /// @param tid Tensor ID in the map. 
      /// @return Unique pointer to tensor with given ID. 
      qtnh::tptr extract(qtnh::uint tid);
      /// @brief Insert tensor in the map. 
      /// @param tp Unique pointer to the tensor to insert. 
      /// @return ID of inserted tensor. 
      qtnh::uint insert(qtnh::tptr tp);
      
      /// @brief Construct a tensor directly inside the tensor network. 
      /// @tparam T Derived tensor class to call the constructor of. 
      /// @tparam ...U Constructor argument types. 
      /// @param ...us Constructor arguments. 
      /// @return ID of constructed tensor. 
      template<class T, class... U>
      qtnh::uint make(U&&... us) {
        tensors_.insert({ ++tensor_counter, T::make(std::forward<U>(us)...) });
        return tensor_counter;
      }

      /// @brief Create bond between two tensors in the tensor network. 
      /// @param tid1 First tensor ID. 
      /// @param tid2 Second tensor ID. 
      /// @param ws Wires between given tensors. 
      /// @return ID of created bond. 
      qtnh::uint addBond(qtnh::uint tid1, qtnh::uint tid2, std::vector<qtnh::wire> ws);

      /// @brief Contract bond with ID. 
      /// @param bid ID of the bond to be contracted. 
      /// @return ID of tensor created by the contraction. 
      ///
      /// Tensors contracted by the bond get deleted from memory, so
      /// any pointers and references to them are no longer valid. 
      qtnh::uint contractBond(qtnh::uint bid);
      /// @brief Contract all bonds according to arbitrary order. 
      /// @return ID of final tensor in the network. 
      qtnh::uint contractAll();
      /// @brief Contract all bonds ordered by the vector of IDs. 
      /// @param bids Vector of bond IDs to contract in given order. 
      /// @return ID of final tensor in the network. 
      qtnh::uint contractAll(std::vector<qtnh::uint> bids);

      /// Print current tensor network, listing all tensors and bonds in it. 
      void print();

    private:
      inline static qtnh::uint tensor_counter = 0;  ///< Counter to determine tensor IDs. 
      inline static qtnh::uint bond_counter = 0;    ///< Counter to determine bond IDs. 

      /// Map between tensors in the network and their IDs. 
      std::unordered_map<qtnh::uint, qtnh::tptr> tensors_;
      /// Map between bonds in the network and their IDs. 
      std::unordered_map<qtnh::uint, Bond> bonds_;
  };

  namespace ops {
    /// Print bond information via std::cout. 
    std::ostream& operator<<(std::ostream&, const TensorNetwork::Bond&);
  }
}

#endif