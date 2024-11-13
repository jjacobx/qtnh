#ifndef __CON_PAIR_DEFS__
#define __CON_PAIR_DEFS__

#include "con/pair.hpp"

namespace qtnh {
  template<typename T1, typename T2>
  qtnh::tptr PairContractor<T1, T2>::contract() {
    PairContractor<DenseTensor, DenseTensor> dcon (
      Tensor::convert<DenseTensor>(std::move(this->tp1_)), 
      Tensor::convert<DenseTensor>(std::move(this->tp2_)), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;
  }

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