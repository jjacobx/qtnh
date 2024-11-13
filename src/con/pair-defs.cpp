#include <algorithm>
#include <iostream>
#include <numeric>

#include "con/pair-defs.hpp"
#include "core/utils.hpp"
#include "tensor/indexing.hpp"


namespace qtnh {
  template<> qtnh::tptr PairContractor<Tensor, Tensor>::contract() {
    switch (this->tp1_->type()) {
      case TT::rescTensor:
        return _contract_1<RescTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      case TT::symmTensor:
        return _contract_1<SymmTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      case TT::diagTensor:
        return _contract_1<DiagTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      case TT::idenTensor:
        return _contract_1<IdenTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      default:
        return _contract_1<DenseTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
    }
  }

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract() {
    return Tensor::cast<Tensor>(std::move(tp1_));
  }
}
