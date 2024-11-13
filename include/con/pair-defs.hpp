#ifndef __CON_PAIR_DEFS__
#define __CON_PAIR_DEFS__

#include "con/pair.hpp"

namespace qtnh {
  template<typename T1, typename T2>
  qtnh::tptr _contract_2(qtnh::tptr tp1, qtnh::tptr tp2, ConParams& params) {
    PairContractor<T1, T2> dcon(
      Tensor::cast<T1>(std::move(tp1)), 
      Tensor::cast<T2>(std::move(tp2)), 
      params
    );

    auto tp_res = dcon.contract();
    params = dcon.params();

    return tp_res;
  }

  template<typename T1>
  qtnh::tptr _contract_1(qtnh::tptr tp1, qtnh::tptr tp2, ConParams& params) {
    switch (tp2->type()) {
      case TT::rescTensor:
        return _contract_2<T1, RescTensor>(std::move(tp1), std::move(tp2), params);
      case TT::symmTensor:
        return _contract_2<T1, SymmTensor>(std::move(tp1), std::move(tp2), params);
      case TT::diagTensor:
        return _contract_2<T1, DiagTensor>(std::move(tp1), std::move(tp2), params);
      case TT::idenTensor:
        return _contract_2<T1, IdenTensor>(std::move(tp1), std::move(tp2), params);
      default:
        return _contract_2<T1, DenseTensor>(std::move(tp1), std::move(tp2), params);
    }
  }

  template<typename T1, typename T2, typename Enable>
  qtnh::tptr PairContractor<T1, T2, Enable>::contract() {
    PairContractor<DenseTensor, DenseTensor> dcon (
      Tensor::convert<DenseTensor>(std::move(this->tp1_)), 
      Tensor::convert<DenseTensor>(std::move(this->tp2_)), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;
  }

  template<typename T1>
  qtnh::tptr PairContractor<T1, IdenTensor>::contract() {
    return nullptr;
  }

  template<typename T2>
  qtnh::tptr PairContractor<IdenTensor, T2, rm_if_iden<T2>>::contract() {
    return nullptr;
  }

  template<> qtnh::tptr PairContractor<Tensor, Tensor>::contract();

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<DenseTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DenseTensor, DiagTensor>::contract();

  // template<> qtnh::tptr Contractor<SymmTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<SymmTensor, DiagTensor>::contract();

  // template<> qtnh::tptr Contractor<DiagTensor, DenseTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor, SymmTensor>::contract();
  // template<> qtnh::tptr Contractor<DiagTensor, DiagTensor>::contract();
}

#endif