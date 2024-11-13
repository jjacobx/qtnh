#ifndef __CON_SELF_DEFS__
#define __CON_SELF_DEFS__

#include "con/self.hpp"

namespace qtnh {
  template<typename T>
  qtnh::tptr SelfContractor<T>::contract() {
    SelfContractor<DenseTensor> dcon (
      Tensor::toDense(Tensor::cast<DenseTensor>(std::move(this->tp1_))), 
      this->params_
    );

    auto tp_res = dcon.contract();
    this->params_ = dcon.params();

    return tp_res;
  }

  template<> qtnh::tptr SelfContractor<DenseTensor>::contract();
}

#endif