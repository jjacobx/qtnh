#include "core/utils.hpp"
#include "tensor-new/tensor.hpp"
#include "tensor-new/dense.hpp"
#include "tensor-new/indexing.hpp"

namespace qtnh {
  // TODO: Implement this in a separate file using enums
  std::unique_ptr<Tensor> Tensor::contract(std::unique_ptr<Tensor> t1u, std::unique_ptr<Tensor> t2u, const std::vector<qtnh::wire>& ws) {
    // Validate contraction dimensions
    for (auto& w : ws) {
      if (t1u->totDims().at(w.first) != t2u->totDims().at(w.second)) {
        throw std::invalid_argument("Incompatible contraction dimensions.");
      }
    }

    if (t1u->type() == TT::denseTensor && t1u->type() == TT::denseTensor) {

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

  DenseTensor* _contract_dense(Tensor* t1p, Tensor* t2p, std::vector<qtnh::wire> ws) {
    std::vector<TIFlag> ifls1(t1p->disDims().size(), { "distributed", 0 });
    std::vector<TIFlag> ifls1_loc(t1p->locDims().size(), { "local", 0 });
    ifls1.insert(ifls1.end(), ifls1_loc.begin(), ifls1_loc.end());

    std::vector<TIFlag> ifls2(t2p->disDims().size(), { "distributed", 0 });
    std::vector<TIFlag> ifls2_loc(t2p->locDims().size(), { "local", 0 });
    ifls2.insert(ifls2.end(), ifls2_loc.begin(), ifls2_loc.end());

    for (std::size_t i = 0; i < ws.size(); ++i) {
      ifls1.at(ws.at(i).first) = { "closed", i };
      ifls2.at(ws.at(i).second) = { "closed", i };
    }

    TIndexing ti1(t1p->totDims(), ifls1);
    TIndexing ti2(t1p->totDims(), ifls2);
    auto ti3 = TIndexing::app(ti1, ti2).cut("closed");

    std::size_t loc_size;
    if (t1p->bc().active && t2p->bc().active) {
      loc_size = utils::dims_to_size(ti3.cut("distributed").dims());
    } else {
      loc_size = 0;
    }

    auto els = std::vector<qtnh::tel>(loc_size);
    DenseTensor* t3p = new DenseTensor(t1p->bc().env, ti3.cut("distributed").dims(), ti3.cut("local").dims(), std::move(els), { t1p->bc().off, 1, 1 });

    if (!t3p->bc().active) return;
    
    auto it3 = ti3.num("open").begin();
    for (auto idxs1 : ti1.tup("open")) {
      for (auto idxs2 : ti2.tup("open")) {
        qtnh::tel el3 = 0.0;

        auto it1 = ti1.num("closed", idxs1);
        auto it2 = ti2.num("closed", idxs2);
        while(it1 != it1.end() && it2 != it2.end()) {
          el3 += (*t1p)[*(it1++)] * (*t2p)[*(it2++)];
        }

        (*t3p)[*(it3++)] = el3;
      }
    }
  }
}
