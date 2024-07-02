#include "core/utils.hpp"
#include "tensor/base2.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  // TODO: Implement this in a separate file using enums
  std::unique_ptr<Tensor> Tensor::contract(std::unique_ptr<Tensor> t1u, std::unique_ptr<Tensor> t2u, const std::vector<qtnh::wire>& ws) {
    // Validate contraction dimensions
    for (auto& w : ws) {
      if (t1u->totDims().at(w.first) != t2u->totDims().at(w.second)) {
        throw std::invalid_argument("Incompatible contraction dimensions.");
      }
    }

    // auto* tp = t2u->contract_disp(t1u.get(), ws);
    
    // // Check if one of the input objects is returned
    // if (tp == t1u.get()) {
    //   return t1u;
    // }
    // if (tp == t2u.get()) {
    //   return t2u;
    // }
    //
    // return std::unique_ptr<Tensor>(tp);

    return std::move(t1u);
  }
}
