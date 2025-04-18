#ifndef __TEN_CON_PAIR_DEFS__
#define __TEN_CON_PAIR_DEFS__

#include "ten/con/pair.hpp"

namespace qtnh {
  using pcon = PairContractor<Tensor, Tensor>;

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

  template<typename T1, typename T2>
  qtnh::tptr _contract(qtnh::tptr tp1, qtnh::tptr tp2, ConParams& params) {
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
  qtnh::tptr _contract_disp(qtnh::tptr tp1, qtnh::tptr tp2, ConParams& params) {
    switch (tp2->type()) {
      case TT::rescTensor:
        return _contract<T1, RescTensor>(std::move(tp1), std::move(tp2), params);
      case TT::symmTensor:
        return _contract<T1, SymmTensor>(std::move(tp1), std::move(tp2), params);
      case TT::swapTensor:
        return _contract<T1, SwapTensor>(std::move(tp1), std::move(tp2), params);
      case TT::diagTensor:
        return _contract<T1, DiagTensor>(std::move(tp1), std::move(tp2), params);
      case TT::idenTensor:
        return _contract<T1, IdenTensor>(std::move(tp1), std::move(tp2), params);
      default:
        tp2 = Tensor::convert<DenseTensor>(std::move(tp2));
        return _contract<T1, DenseTensor>(std::move(tp1), std::move(tp2), params);
    }
  }

  template<typename T1, typename T2>
  qtnh::tptr PairContractor<T1, T2, keep_if_base<T1, T2>>::contract() {
    switch (this->tp1_->type()) {
      case TT::rescTensor:
        return _contract_disp<RescTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      case TT::symmTensor:
        return _contract_disp<SymmTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      case TT::swapTensor:
        return _contract_disp<SwapTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      case TT::diagTensor:
        return _contract_disp<DiagTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      case TT::idenTensor:
        return _contract_disp<IdenTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
      default:
        this->tp1_ = Tensor::convert<DenseTensor>(std::move(this->tp1_));
        return _contract_disp<DenseTensor>(std::move(this->tp1_), std::move(this->tp2_), this->params_);
    }
  }

  template<typename T1>
  qtnh::tptr PairContractor<T1, IdenTensor>::contract() {
    // * For now convert to symmetric tensor. 
    PairContractor<T1, SymmTensor> dcon(
      std::move(this->tp1_), 
      Tensor::convert<SymmTensor>(std::move(this->tp2_)), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;

    // TODO: Optimise if all wires match
    // return std::move(this->tp1_);
  }

  template<typename T2>
  qtnh::tptr PairContractor<IdenTensor, T2, rm_if_iden<T2>>::contract() {
    // * For now convert to symmetric tensor. 
    PairContractor<SymmTensor, T2> dcon(
      Tensor::convert<SymmTensor>(std::move(this->tp1_)), 
      std::move(this->tp2_), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;

    // TODO: Optimise if all wires match
    // return std::move(this->tp2_);
  }

  template<typename T1>
  qtnh::tptr PairContractor<T1, SwapTensor, rm_if_iden<T1>>::contract() {
    // * For now convert to symmetric tensor. 
    PairContractor<T1, SymmTensor> dcon(
      std::move(this->tp1_), 
      Tensor::convert<SymmTensor>(std::move(this->tp2_)), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;

    // TODO: Optimise if all wires match
    // auto w1 = this->params_.wires[0].first;
    // auto w2 = this->params_.wires[1].first;
    // return Tensor::swap(std::move(this->tp1_), w1, w2);
  }

  // template<> qtnh::tptr PairContractor<Tensor, Tensor>::contract();

  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract();
  template<> qtnh::tptr PairContractor<DenseTensor, SymmTensor>::contract();
  // template<> qtnh::tptr PairContractor<DenseTensor, DiagTensor>::contract();

  // template<> qtnh::tptr PairContractor<SymmTensor, DenseTensor>::contract();
  // template<> qtnh::tptr PairContractor<SymmTensor, SymmTensor>::contract();
  // template<> qtnh::tptr PairContractor<SymmTensor, DiagTensor>::contract();

  // template<> qtnh::tptr PairContractor<DiagTensor, DenseTensor>::contract();
  // template<> qtnh::tptr PairContractor<DiagTensor, SymmTensor>::contract();
  // template<> qtnh::tptr PairContractor<DiagTensor, DiagTensor>::contract();
}

#endif