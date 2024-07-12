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
    std::vector<TIFlag> ifls1(t1p->totDims().size(), { "open", 0 });
    std::vector<TIFlag> ifls2(t2p->totDims().size(), { "open", 0 });
    for (std::size_t i = 0; i < ws.size(); ++i) {
      ifls1.at(ws.at(i).first) = { "closed", i };
      ifls2.at(ws.at(i).first) = { "closed", i };
    }

    TIndexing ti1(t1p->totDims(), ifls1);
    TIndexing ti2(t1p->totDims(), ifls2);
    auto ti3 = TIndexing::app(ti1, ti2);

    for (auto idxs1 : ti1.tup("open", qtnh::tidx_tup(t1p->totDims().size(), 0))) {
      for (auto idxs2 : ti2.tup("open", qtnh::tidx_tup(t2p->totDims().size(), 0))) {
        
      }
    }
  }
}
