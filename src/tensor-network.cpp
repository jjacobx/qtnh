#include <iostream>
#include <mpi.h>

#include "tensor/network.hpp"

namespace qtnh {
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
    for (auto i = 0; i < dist_size1; ++i) {
      if (is_open1.at(i)) t1_imaps.insert({i, counter++});
    }
    for (auto i = 0; i < dist_size2; ++i) {
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
}
