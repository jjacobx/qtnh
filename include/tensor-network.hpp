#ifndef TENSOR_NETWORK_HPP
#define TENSOR_NETWORK_HPP

#include "dense-tensor.hpp"

namespace qtnh {
  struct Bond {
    private:
      inline static qtnh::uint counter = 0;
      const qtnh::uint id;
    
    public:
      std::pair<qtnh::uint, qtnh::uint> tensor_ids;
      std::vector<qtnh::wire> wires;

      Bond(std::pair<qtnh::uint, qtnh::uint> tensor_ids, std::vector<qtnh::wire> wires)
        : id(++counter), tensor_ids(tensor_ids), wires(wires) {}
      ~Bond() = default;
      
      qtnh::uint getID() { return id; }
  };

  class TensorNetwork {
    private:
      std::map<qtnh::uint, Tensor&> tensors;
      std::map<qtnh::uint, Bond&> bonds;

    public:
      TensorNetwork()
        : tensors(std::map<qtnh::uint, Tensor&>()), bonds(std::map<qtnh::uint, Bond&>()) {}
      TensorNetwork(const TensorNetwork&) = delete;
      ~TensorNetwork() = default;

      Tensor& getTensor(qtnh::uint k) { return tensors.at(k); }
      Bond& getBond(qtnh::uint k) { return bonds.at(k); }

      void insertTensor(Tensor& t) { tensors.insert({t.getID(), t}); return; }
      void insertBond(Bond& b) { bonds.insert({b.getID(), b}); return; }

      qtnh::uint contractBond(qtnh::uint);
      qtnh::uint contractAll();
  };
}

#endif