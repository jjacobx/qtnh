#include <iostream>
#include <mpi.h>

#include "tensor/indexing.hpp"
#include "tensor/network.hpp"

namespace qtnh {
  typedef std::unordered_map<qtnh::tidx_tup_st, qtnh::tidx_tup_st> tidx_tup_imap;
  std::pair<tidx_tup_imap, tidx_tup_imap> _get_timaps(const Tensor& t1, const Tensor& t2, std::vector<qtnh::wire> ws, bool in_place) {
    std::vector<bool> is_open1(t1.getDims().size(), true);
    std::vector<bool> is_open2(t2.getDims().size(), true);

    for (auto w : ws) {
      is_open1.at(w.first) = false;
      is_open2.at(w.second) = false;
    }

    qtnh::tidx_tup_st counter = 0;
    auto t1_imaps = tidx_tup_imap();
    auto t2_imaps = tidx_tup_imap();

    auto i2 = 0; // index for in-place contractions

    for (std::size_t i = 0; i < t1.getDistDims().size(); ++i) {
      if (is_open1.at(i))  {
        t1_imaps.insert({ i, counter++ });
      } else if (in_place) {
        // If in place, replace closed indices of t1 with open indices of t2. 
        while (!is_open2.at(i2)) ++i2; // find next open index of t2
        t2_imaps.insert({ i2++, counter++ });
      }
    }

    // If not in-place, append distributed indices sequentially. 
    for (std::size_t i = 0; !in_place && i < t2.getDistDims().size(); ++i) {
      if (is_open2.at(i)) t2_imaps.insert({ i, counter++ });
    }

    for (auto i = t1.getDistDims().size(); i < t1.getDims().size(); ++i) {
      if (is_open1.at(i)) { 
        t1_imaps.insert({ i, counter++ });
      } else if (in_place) {
        // If in place, replace closed indices of t1 with open indices of t2. 
        while (!is_open2.at(i2)) ++i2; // find next open index of t2
        t2_imaps.insert({ i2++, counter++ });
      }
    }

    // If not in-place, append local indices sequentially. 
    for (auto i = t2.getDistDims().size(); !in_place && i < t2.getDims().size(); ++i) {
      if (is_open2.at(i)) t2_imaps.insert({ i, counter++ });
    }

    return { t1_imaps, t2_imaps };
  }

  TensorNetwork::TensorNetwork() : 
    tensors(std::unordered_map<qtnh::uint, std::unique_ptr<Tensor>>()), 
    bonds(std::unordered_map<qtnh::uint, Bond>()) {}
  
  TensorNetwork::Bond::Bond(std::pair<qtnh::uint, qtnh::uint> t_ids, std::vector<qtnh::wire> ws, bool in_place) : 
    tensor_ids(t_ids), wires(ws), in_place(in_place) {}
  
  Tensor& TensorNetwork::getTensor(qtnh::uint k) {
    return *tensors.at(k).get();
  }

  std::unique_ptr<Tensor> TensorNetwork::extractTensor(qtnh::uint id) {
    auto tu = std::move(tensors.at(id));
    tensors.erase(id);

    return tu;
  }

  std::vector<qtnh::uint> TensorNetwork::getTensorIDs() {
    std::vector<qtnh::uint> tensor_ids;
    for (auto& [k, v] : tensors) {
      tensor_ids.push_back(k);
    }

    return tensor_ids;
  }

  TensorNetwork::Bond TensorNetwork::getBond(qtnh::uint k) { 
    return bonds.at(k);
  }

  qtnh::uint TensorNetwork::createBond(qtnh::uint t1_id, qtnh::uint t2_id, std::vector<qtnh::wire> ws, bool in_place) {
    Bond b({ t1_id, t2_id }, ws, in_place);
    bonds.insert({ ++bond_counter, b });

    return bond_counter;
  }

  qtnh::uint TensorNetwork::insertTensor(std::unique_ptr<Tensor> tu) {
    tensors.insert({ ++tensor_counter, std::move(tu) }); 
    return tensor_counter; 
  }

  qtnh::uint TensorNetwork::contractBond(qtnh::uint id) {
    auto& b = bonds.at(id);
    auto [t1_id, t2_id] = b.tensor_ids;

    auto t1_up = std::move(tensors.at(t1_id));
    auto t2_up = std::move(tensors.at(t2_id));

    auto [t1_imaps, t2_imaps] = _get_timaps(*t1_up, *t2_up, b.wires, b.in_place);
    auto t3_p = Tensor::contract(std::move(t1_up), std::move(t2_up), b.wires);

    bonds.erase(id);
    tensors.erase(t1_id);
    tensors.erase(t2_id);

    auto t3_id = insertTensor(std::move(t3_p));
    
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
      auto b1_id = bonds_order.at(i);
      auto& b1 = bonds.at(b1_id); // must be a reference, as it is later updated

      for (j = 1; i + j < bonds_order.size(); ++j) {
        auto b2_id = bonds_order.at(i + j);
        auto b2 = bonds.at(b2_id);
        if (b1.tensor_ids.first == b2.tensor_ids.first && b1.tensor_ids.second == b2.tensor_ids.second) {
          b1.wires.insert(b1.wires.end(), b2.wires.begin(), b2.wires.end());
          bonds.erase(b2_id);
        } else {
          break;
        }
      }

      #ifdef DEBUG
        using namespace qtnh::ops;
        std::cout << "Contracting " << bonds.at(b1_id) << "\n";
      #endif
      tid = contractBond(b1_id);

      #ifdef DEBUG
        int proc_id;
        MPI_Comm_rank(MPI_COMM_WORLD, &proc_id);
        if (proc_id == 0) print();
      #endif
    }

    return tid;
  }

  void TensorNetwork::print() {
    using namespace qtnh::ops;

    std::cout << "================================================================\n";
    std::cout << "Tensor Network of " << tensors.size() << " tensors and " << bonds.size() << " bonds\n";

    std::cout << "----------------------------------------------------------------\n";
    std::cout << "Tensors: \n";
    for (auto& [id, t] : tensors) {
      std::cout << "ID: " << id << " | Els: " << *t.get() << "\n";
    }

    std::cout << "----------------------------------------------------------------\n";
    std::cout << "Bonds: \n";
    for (auto& [id, b] : bonds) {
      std::cout << "ID: " << id << " | Bond: " << b << "\n";
    }

    std::cout << "================================================================\n";
  }

  std::ostream& ops::operator<<(std::ostream& out, const TensorNetwork::Bond& o) {
    out << "(" << o.tensor_ids.first << ", " << o.tensor_ids.second << "); ";
    out << "{";
    for (std::size_t i = 0; i < o.wires.size(); ++i) {
      out << "(" << o.wires.at(i).first << ", " << o.wires.at(i).second << ")";
      if (i < o.wires.size() - 1) out << ", ";
    }

    out << "}";
    return out;
  }
}