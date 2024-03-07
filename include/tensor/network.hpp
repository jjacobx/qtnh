#ifndef _TENSOR__NETWORK_HPP
#define _TENSOR__NETWORK_HPP

// #include <map>
#include <memory>
#include <unordered_map>

#include "../core/typedefs.hpp"
#include "tensor/dense.hpp"

namespace qtnh {
  /// Storage for tensors and bonds connecting them. 
  class TensorNetwork {
    public:
      /// Create empty tensor network. 
      TensorNetwork();
      /// Copy constructor is invalid since tensors should not be in multiple networks at once. 
      TensorNetwork(const TensorNetwork&) = delete;

      /// Default destructor. 
      ~TensorNetwork() = default;

      /// Used for storage of wires for tensor contraction, together with their two target tensors. 
      struct Bond {
        std::pair<qtnh::uint, qtnh::uint> tensor_ids;  ///< IDs of two target tensors. 
        std::vector<qtnh::wire> wires;                 ///< Wires connecting tensor indices. 

        /// @brief General constructor of a bond. 
        /// @param t_ids Pair of IDs of target tensors. 
        /// @param ws Vector of wires for contraction. 
        Bond(std::pair<qtnh::uint, qtnh::uint> t_ids, std::vector<qtnh::wire> ws);

        /// Default destructor. 
        ~Bond() = default;
      };

      /// @brief Get tensor with ID. 
      /// @param id Tensor ID in the map. 
      /// @return Reference to the tensor with given ID. 
      Tensor& getTensor(qtnh::uint id);
      /// @brief Get bond with ID. 
      /// @param id Bond ID in the map. 
      /// @return Copy of the bond with given ID. 
      Bond getBond(qtnh::uint id);

      /// @brief Construct a tensor directly inside the tensor network. 
      /// @tparam T Derived tensor class to call the constructor of. 
      /// @tparam ...U Constructor argument types. 
      /// @param ...us Constructor arguments. 
      /// @return ID of constructed tensor. 
      template<class T, class... U>
      qtnh::uint createTensor(U&&... us) {
        tensors.insert({ ++tensor_counter, std::make_unique<T>(std::forward<U>(us)...) });
        return tensor_counter;
      }

      /// @brief Create bond between two tensors in the tensor network. 
      /// @param t1_id First tensor ID. 
      /// @param t2_id Second tensor ID. 
      /// @param ws Wires between given tensors. 
      /// @return ID of created bond. 
      qtnh::uint createBond(qtnh::uint t1_id, qtnh::uint t2_id, std::vector<qtnh::wire> ws);

      /// @brief Insert tensor in the map. 
      /// @param t Reference to tensor to insert. 
      /// @return ID of inserted tensor. 
      qtnh::uint insertTensor(std::unique_ptr<Tensor> tu);

      /// @brief Contract bond with ID. 
      /// @param id ID of the bond to be contracted. 
      /// @return ID of tensor created by the contraction. 
      ///
      /// Tensors contracted by the bond get deleted from memory, so
      /// any pointers and references to them are no longer valid. 
      qtnh::uint contractBond(qtnh::uint id);
      /// @brief Contract all bonds according to arbitrary order. 
      /// @return ID of final tensor in the network. 
      qtnh::uint contractAll();
      /// @brief Contract all bonds ordered by the vector of IDs. 
      /// @param ids Vector of bond IDs to contract in given order. 
      /// @return ID of final tensor in the network. 
      qtnh::uint contractAll(std::vector<qtnh::uint> ids);

      /// Print current tensor network, listing all tensors and bonds in it. 
      void print();

    private:
      inline static qtnh::uint tensor_counter = 0;  ///< Counter to determine tensor IDs. 
      inline static qtnh::uint bond_counter = 0;    ///< Counter to determine bond IDs. 

      /// Map between tensors in the network and their IDs. 
      std::unordered_map<qtnh::uint, std::unique_ptr<Tensor>> tensors;
      /// Map between bonds in the network and their IDs. 
      std::unordered_map<qtnh::uint, Bond> bonds;
  };

  namespace ops {
    /// Print bond information via std::cout. 
    std::ostream& operator<<(std::ostream&, const TensorNetwork::Bond&);
  }
}

#endif