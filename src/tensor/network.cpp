#include <iostream>
#include <mpi.h>

#include "tensor/indexing.hpp"
#include "tensor/network.hpp"

namespace qtnh {
  Bond::Bond(std::pair<qtnh::uint, qtnh::uint> tensor_ids, std::vector<qtnh::wire> wires)
    : id(++counter), tensor_ids(tensor_ids), wires(wires) {}

  std::ostream& ops::operator<<(std::ostream& out, const Bond& o) {
    out << "(" << o.tensor_ids.first << ", " << o.tensor_ids.second << "); ";
    out << "{";
    for (std::size_t i = 0; i < o.wires.size(); ++i) {
      out << "(" << o.wires.at(i).first << ", " << o.wires.at(i).second << ")";
      if (i < o.wires.size() - 1) out << ", ";
    }

    out << "}";
    return out;
  }

  TensorNetwork::TensorNetwork()
    : tensors(std::map<qtnh::uint, Tensor&>()), bonds(std::map<qtnh::uint, Bond&>()) {}
  
  Tensor& TensorNetwork::getTensor(qtnh::uint k) { 
    return tensors.at(k);
  }

  Bond& TensorNetwork::getBond(qtnh::uint k) { 
    return bonds.at(k);
  }

  qtnh::uint TensorNetwork::insertTensor(Tensor& t) { 
    tensors.insert({t.getID(), t}); 
    return t.getID(); 
  }

  qtnh::uint TensorNetwork::insertBond(Bond& b) { 
    bonds.insert({b.getID(), b}); 
    return b.getID(); 
  }

  qtnh::uint TensorNetwork::contractBond(qtnh::uint id) {
    auto& b = bonds.at(id);
    auto& t1 = tensors.at(b.tensor_ids.first);
    auto& t2 = tensors.at(b.tensor_ids.second);

    auto dims1 = t1.getDims();
    auto dims2 = t2.getDims();
    auto dist_size1 = t1.getDistDims().size();
    auto dist_size2 = t2.getDistDims().size();

    std::vector<bool> is_open1(dims1.size(), true);
    std::vector<bool> is_open2(dims2.size(), true);

    for (auto w : b.wires) {
      is_open1.at(w.first) = false;
      is_open2.at(w.second) = false;
    }

    qtnh::tidx_tup_st counter = 0;
    auto t1_imaps = std::map<qtnh::tidx_tup_st, qtnh::tidx_tup_st>();
    auto t2_imaps = std::map<qtnh::tidx_tup_st, qtnh::tidx_tup_st>();
    for (std::size_t i = 0; i < dist_size1; ++i) {
      if (is_open1.at(i)) t1_imaps.insert({i, counter++});
    }
    for (std::size_t i = 0; i < dist_size2; ++i) {
      if (is_open2.at(i)) t2_imaps.insert({i, counter++});
    }
    for (auto i = dist_size1; i < dims1.size(); ++i) {
      if (is_open1.at(i)) t1_imaps.insert({i, counter++});
    }
    for (auto i = dist_size2; i < dims2.size(); ++i) {
      if (is_open2.at(i)) t2_imaps.insert({i, counter++});
    }

    auto* t3r = Tensor::contract(&t1, &t2, b.wires);

    bonds.erase(b.getID());
    tensors.erase(t1.getID());
    tensors.erase(t2.getID());
    tensors.insert({t3r->getID(), *t3r});

    for (auto kv : bonds) {
      auto& b = kv.second;
      if (b.tensor_ids.first == t1.getID()) {
        b.tensor_ids.first = t3r->getID();
        for (auto& w : b.wires) {
          w.first = t1_imaps.at(w.first);
        }
      } else if (b.tensor_ids.first == t2.getID()) {
        b.tensor_ids.first = t3r->getID();
        for (auto& w : b.wires) {
          w.first = t2_imaps.at(w.first);
        }
      }

      if (b.tensor_ids.second == t1.getID()) {
        b.tensor_ids.second = t3r->getID();
        for (auto& w : b.wires) {
          w.second = t1_imaps.at(w.second);
        }
      } else if (b.tensor_ids.second == t2.getID()) {
        b.tensor_ids.second = t3r->getID();
        for (auto& w : b.wires) {
          w.second = t2_imaps.at(w.second);
        }
      }
    }

    return t3r->getID();
  }

  qtnh::uint TensorNetwork::contractAll() {
    auto id = (*tensors.begin()).first;
    auto temp_bonds = bonds;
    for (auto kv : temp_bonds) {
      id = contractBond(kv.first);
    }

    return id;
  }

  qtnh::uint TensorNetwork::contractAll(std::vector<qtnh::uint> bonds_order) {
    auto tid = (*tensors.begin()).first;
    for (std::size_t i = 0, j = 1; i < bonds_order.size(); i += j) {
      auto bid = bonds_order.at(i);
      auto& b1 = bonds.at(bid);

      for (j = 1; i + j < bonds_order.size(); ++j) {
        auto& b2 = bonds.at(bonds_order.at(i + j));
        if (b1.tensor_ids.first == b2.tensor_ids.first && b1.tensor_ids.second == b2.tensor_ids.second) {
          b1.wires.insert(b1.wires.end(), b2.wires.begin(), b2.wires.end());
          bonds.erase(b2.getID());
        } else {
          break;
        }
      }

      tid = contractBond(bid);

      #ifdef DEBUG
        print();
      #endif
    }

    return tid;
  }

  void TensorNetwork::print() {
    using namespace qtnh::ops;

    std::cout << "================================================================" << std::endl;
    std::cout << "Tensor Network of " << tensors.size() << " tensors and " << bonds.size() << " bonds" << std::endl;

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Tensors: " << std::endl;
    for (auto kv : tensors) {
      std::cout << "ID: " << kv.first << " | Els: " << kv.second << std::endl;
    }

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Bonds: " << std::endl;
    for (auto kv : bonds) {
      std::cout << "ID: " << kv.first << " | Bond: " << kv.second << std::endl;
    }

    std::cout << "================================================================" << std::endl;
  }
}
