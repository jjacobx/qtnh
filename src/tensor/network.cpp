#include <iostream>
#include <mpi.h>

#include "tensor/indexing.hpp"
#include "tensor/network.hpp"

namespace qtnh {
  TensorNetwork::TensorNetwork() : 
    tensors_(std::unordered_map<qtnh::uint, std::unique_ptr<Tensor>>()), 
    bonds_(std::unordered_map<qtnh::uint, Bond>()) {}
  
  TensorNetwork::Bond::Bond(std::pair<qtnh::uint, qtnh::uint> tids, std::vector<qtnh::wire> ws) : 
    tensor_ids(tids), wires(ws) {}
  
  Tensor* TensorNetwork::tensor(qtnh::uint tid) {
    return tensors_.at(tid).get();
  }

  const TensorNetwork::Bond& TensorNetwork::bond(qtnh::uint bid) {
    return bonds_.at(bid);
  }

  std::vector<qtnh::uint> TensorNetwork::tensorIDs() {
    std::vector<qtnh::uint> tensor_ids;
    for (auto& [k, v] : tensors_) {
      tensor_ids.push_back(k);
    }

    return tensor_ids;
  }

  std::vector<qtnh::uint> TensorNetwork::bondIDs() {
    std::vector<qtnh::uint> tensor_ids;
    for (auto& [k, v] : bonds_) {
      tensor_ids.push_back(k);
    }

    return tensor_ids;
  }

  std::unique_ptr<Tensor> TensorNetwork::extract(qtnh::uint tid) {
    auto tp = std::move(tensors_.at(tid));
    tensors_.erase(tid);

    return tp;
  }

  qtnh::uint TensorNetwork::insert(qtnh::tptr tp) {
    tensors_.insert({ ++tensor_counter, std::move(tp) }); 
    return tensor_counter; 
  }

  qtnh::uint TensorNetwork::addBond(qtnh::uint tid1, qtnh::uint tid2, std::vector<qtnh::wire> ws) {
    Bond b({ tid1, tid2 }, ws);
    bonds_.insert({ ++bond_counter, b });

    return bond_counter;
  }

  qtnh::uint TensorNetwork::contractBond(qtnh::uint bid) {
    auto& b = bonds_.at(bid);
    auto [tid1, tid2] = b.tensor_ids;

    auto tp1 = std::move(tensors_.at(tid1));
    auto tp2 = std::move(tensors_.at(tid2));

    ConParams params(b.wires);
    auto tp3 = Tensor::contract(std::move(tp1), std::move(tp2), params);

    auto dim_repls1 = params.dimRepls1;
    auto dim_repls2 = params.dimRepls2;

    bonds_.erase(bid);
    tensors_.erase(tid1);
    tensors_.erase(tid2);

    auto tid3 = insert(std::move(tp3));

    for (auto& [id, b] : bonds_) {
      if (b.tensor_ids.first == tid1) {
        b.tensor_ids.first = tid3;
        for (auto& w : b.wires) {
          w.first = dim_repls1.at(w.first);
        }
      } else if (b.tensor_ids.first == tid2) {
        b.tensor_ids.first = tid3;
        for (auto& w : b.wires) {
          w.first = dim_repls2.at(w.first);
        }
      }

      if (b.tensor_ids.second == tid1) {
        b.tensor_ids.second = tid3;
        for (auto& w : b.wires) {
          w.second = dim_repls1.at(w.second);
        }
      } else if (b.tensor_ids.second == tid2) {
        b.tensor_ids.second = tid3;
        for (auto& w : b.wires) {
          w.second = dim_repls2.at(w.second);
        }
      }
    }

    return tid3;
  }

  qtnh::uint TensorNetwork::contractAll() {
    auto tid = (*tensors_.begin()).first;
    auto temp_bonds = bonds_;
    for (auto& [bid, b] : temp_bonds) {
      #ifdef DEBUG
        utils::barrier();
        if (utils::is_root()) {
          using namespace qtnh::ops;
          std::cout << "Contracting " << b << " in the following tensor network: \n";
          print();
        }
        utils::barrier();
      #endif

      tid = contractBond(bid);
    }

    return tid;
  }

  qtnh::uint TensorNetwork::contractAll(std::vector<qtnh::uint> bonds_order) {
    auto tid = (*tensors_.begin()).first;
    for (std::size_t i = 0, j = 1; i < bonds_order.size(); i += j) {
      auto bid1 = bonds_order.at(i);
      auto& b1 = bonds_.at(bid1); // must be a reference, as it is later updated

      for (j = 1; i + j < bonds_order.size(); ++j) {
        auto bid2 = bonds_order.at(i + j);
        auto b2 = bonds_.at(bid2);
        if (b1.tensor_ids.first == b2.tensor_ids.first && b1.tensor_ids.second == b2.tensor_ids.second) {
          b1.wires.insert(b1.wires.end(), b2.wires.begin(), b2.wires.end());
          bonds_.erase(bid2);
        } else {
          break;
        }
      }

      #ifdef DEBUG
        utils::barrier();
        if (utils::is_root()) {
          using namespace qtnh::ops;
          std::cout << "Contracting " << bonds_.at(bid1) << " in the following tensor network: \n";
          print();
        }
        utils::barrier();
      #endif

      tid = contractBond(bid1);
      tensors_.at(tid) = Tensor::rebcast(std::move(tensors_.at(tid)), { 1, 1, 0 });

      #ifdef DEBUG
        utils::barrier();
        auto& t = *tensors_.at(tid);
        if (t.bc().active) {
          using namespace ops;
          std::cout << t.bc().env.proc_id << " | T (result) = " << t << "\n";
        }
        utils::barrier();
      #endif
    }

    return tid;
  }

  void TensorNetwork::print() {
    using namespace qtnh::ops;

    std::cout << "================================================================\n";
    std::cout << "Tensor Network of " << tensors_.size() << " tensors and " << bonds_.size() << " bonds\n";

    std::cout << "----------------------------------------------------------------\n";
    std::cout << "Tensors: \n";
    for (auto& [id, tp] : tensors_) {
      std::cout << "ID: " << id << " | Els: " << *tp.get() << "\n";
    }

    std::cout << "----------------------------------------------------------------\n";
    std::cout << "Bonds: \n";
    for (auto& [id, b] : bonds_) {
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
