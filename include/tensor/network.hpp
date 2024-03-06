#ifndef _TENSOR__NETWORK_HPP
#define _TENSOR__NETWORK_HPP

#include <map>
#include <memory>

#include "../core/typedefs.hpp"
#include "tensor/dense.hpp"

namespace qtnh {
  /// Used for storage of wires for tensor contraction, together with their two target tensors. 
  struct Bond {
    private:
      inline static qtnh::uint counter = 0;  ///< Counter of created bonds. 
      const qtnh::uint id;                   ///< ID of current bond. 
    
    public:
      std::pair<qtnh::uint, qtnh::uint> tensor_ids;  ///< IDs of two target tensors. 
      std::vector<qtnh::wire> wires;                 ///< Wires connecting tensor indices. 

      /// @brief General constructor of a bond. 
      /// @param t_ids Pair of IDs of target tensors. 
      /// @param ws Vector of wires for contraction. 
      Bond(std::pair<qtnh::uint, qtnh::uint> t_ids, std::vector<qtnh::wire> ws);

      /// Default destructor. 
      ~Bond() = default;
      
      qtnh::uint getID() { return id; }  /// Bond ID getter. 
  };

  namespace ops {
    /// Print bond information via std::cout. 
    std::ostream& operator<<(std::ostream&, const Bond&);
  }

  /// Storage for tensors and bonds connecting them. 
  class TensorNetwork {
    private:
      inline static qtnh::uint tensor_counter = 0;  ///< Counter to determine tensor IDs. 
      inline static qtnh::uint bond_counter = 0;    ///< Counter to determine bond IDs. 

      /// Map between tensors in the network and their IDs. 
      std::unordered_map<qtnh::uint, std::unique_ptr<Tensor>> tensors;
      /// Map between bonds in the network and their IDs. 
      std::unordered_map<qtnh::uint, Bond&> bonds;                      

    public:
      /// Create empty tensor network. 
      TensorNetwork();
      /// Copy constructor is invalid since tensors should not be in multiple networks at once. 
      TensorNetwork(const TensorNetwork&) = delete;

      /// Default destructor. 
      ~TensorNetwork() = default;

      /// @brief Get tensor with ID. 
      /// @param id Tensor ID in the map. 
      /// @return Reference to the tensor with given ID. 
      Tensor& getTensor(qtnh::uint id);
      /// @brief Get bond with ID. 
      /// @param id Bond ID in the map. 
      /// @return Reference to the bond with given ID. 
      Bond& getBond(qtnh::uint id);

      template<class T, class... U>
      qtnh::uint createTensor(U&&... us) {
        tensors.insert({ ++tensor_counter, std::unique_ptr<Tensor>(new T(std::forward<U>(us)...)) });
        return tensor_counter;
      }
      /// @brief Insert tensor in the map. 
      /// @param t Reference to tensor to insert. 
      /// @return ID of the inserted tensor. 
      qtnh::uint insertTensor(Tensor* t);
      /// @brief Insert bond in the map. 
      /// @param b Reference to bond to insert. 
      /// @return ID of the inserted bond. 
      qtnh::uint insertBond(Bond& b);

      /// @brief Contract bond with ID. 
      /// @param id ID of the bond to be contracted. 
      /// @return ID of the tensor created by the contraction. 
      qtnh::uint contractBond(qtnh::uint id);
      /// @brief Contract all bonds according to arbitrary order. 
      /// @return ID of the final tensor in the network. 
      qtnh::uint contractAll();
      /// @brief Contract all bonds ordered by the vector of IDs. 
      /// @param ids Vector of bond IDs to contract in given order. 
      /// @return ID of the final tensor in the network. 
      qtnh::uint contractAll(std::vector<qtnh::uint> ids);

      /// Print current tensor network, listing all tensors and bonds in it. 
      void print();
  };
}

#endif