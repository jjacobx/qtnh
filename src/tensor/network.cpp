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

  TensorNetwork::TensorNetwork() : 
    tensors(std::unordered_map<qtnh::uint, std::unique_ptr<Tensor>>()), 
    bonds(std::unordered_map<qtnh::uint, Bond&>()) {}
  
  Tensor& TensorNetwork::getTensor(qtnh::uint k) {
    return *tensors.at(k).get();
  }

  Bond& TensorNetwork::getBond(qtnh::uint k) { 
    return bonds.at(k);
  }

  qtnh::uint TensorNetwork::insertTensor(Tensor* t) {
    tensors.insert({ ++tensor_counter, std::unique_ptr<Tensor>(t) }); 
    return tensor_counter; 
  }

  qtnh::uint TensorNetwork::insertBond(Bond& b) { 
    bonds.insert({b.getID(), b}); 
    return b.getID(); 
  }

  qtnh::uint TensorNetwork::contractBond(qtnh::uint id) {
    auto& b = bonds.at(id);
    auto [t1_id, t2_id] = b.tensor_ids;

    auto t1_up = std::move(tensors.at(t1_id));
    auto t2_up = std::move(tensors.at(t2_id));

    auto dims1 = t1_up->getDims();
    auto dims2 = t2_up->getDims();
    auto dist_size1 = t1_up->getDistDims().size();
    auto dist_size2 = t2_up->getDistDims().size();

    std::vector<bool> is_open1(dims1.size(), true);
    std::vector<bool> is_open2(dims2.size(), true);

    for (auto w : b.wires) {
      is_open1.at(w.first) = false;
      is_open2.at(w.second) = false;
    }

    qtnh::tidx_tup_st counter = 0;
    auto t1_imaps = std::unordered_map<qtnh::tidx_tup_st, qtnh::tidx_tup_st>();
    auto t2_imaps = std::unordered_map<qtnh::tidx_tup_st, qtnh::tidx_tup_st>();
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

    auto t3_p = Tensor::contract(t1_up.get(), t2_up.get(), b.wires);

    bonds.erase(b.getID());
    delete &b;

    tensors.erase(t1_id);
    tensors.erase(t2_id);

    auto t3_id = insertTensor(t3_p);

    for (auto& [id, b] : bonds) {
      if (b.tensor_ids.first == t1_id) {
        b.tensor_ids.first = t3_id;
        for (auto& w : b.wires) {
          w.first = t1_imaps.at(w.first);
        }
      } else if (b.tensor_ids.first == t2_id) {
        b.tensor_ids.first = t3_id;
        for (auto& w : b.wires) {
          w.first = t2_imaps.at(w.first);
        }
      }

      if (b.tensor_ids.second == t1_id) {
        b.tensor_ids.second = t3_id;
        for (auto& w : b.wires) {
          w.second = t1_imaps.at(w.second);
        }
      } else if (b.tensor_ids.second == t2_id) {
        b.tensor_ids.second = t3_id;
        for (auto& w : b.wires) {
          w.second = t2_imaps.at(w.second);
        }
      }
    }

    return t3_id;
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
          delete &b2;
        } else {
          break;
        }
      }

      tid = contractBond(bid);
      std::cout << "CHECKPOINT\n";

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
    for (auto& [id, t] : tensors) {
      std::cout << "ID: " << id << " | Els: " << *t.get() << std::endl;
    }

    std::cout << "----------------------------------------------------------------" << std::endl;
    std::cout << "Bonds: " << std::endl;
    for (auto& [id, b] : bonds) {
      std::cout << "ID: " << id << " | Bond: " << b << std::endl;
    }

    std::cout << "================================================================" << std::endl;
  }
}
