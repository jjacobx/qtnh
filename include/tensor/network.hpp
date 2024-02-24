#ifndef _TENSOR__NETWORK_HPP
#define _TENSOR__NETWORK_HPP

#include <map>

#include "../core/typedefs.hpp"
#include "tensor/dense.hpp"

namespace qtnh {
  struct Bond {
    private:
      inline static qtnh::uint counter = 0;
      const qtnh::uint id;
    
    public:
      std::pair<qtnh::uint, qtnh::uint> tensor_ids;
      std::vector<qtnh::wire> wires;

      Bond(std::pair<qtnh::uint, qtnh::uint> tensor_ids, std::vector<qtnh::wire> wires);
      ~Bond() = default;
      
      qtnh::uint getID() { return id; }
  };

  namespace ops {
    std::ostream& operator<<(std::ostream&, const Bond&);
  }

  class TensorNetwork {
    private:
      std::map<qtnh::uint, Tensor&> tensors;
      std::map<qtnh::uint, Bond&> bonds;

    public:
      TensorNetwork();
      TensorNetwork(const TensorNetwork&) = delete;
      ~TensorNetwork() = default;

      Tensor& getTensor(qtnh::uint k);
      Bond& getBond(qtnh::uint k);

      qtnh::uint insertTensor(Tensor& t);
      qtnh::uint insertBond(Bond& b);

      qtnh::uint contractBond(qtnh::uint);
      qtnh::uint contractAll();
      qtnh::uint contractAll(std::vector<qtnh::uint>);

      void print();
  };
}

#endif