#include <algorithm>
#include <iostream>
#include <numeric>

#include "core/utils.hpp"
#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/symm.hpp"
#include "tensor/diag.hpp"
#include "tensor/indexing.hpp"
#include "contract/general-temp.hpp"

namespace qtnh {
  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract() {
    return Tensor::cast<Tensor>(std::move(tp1_));
  }

  template<> qtnh::tptr SelfContractor<DenseTensor>::contract() {
    return Tensor::cast<Tensor>(std::move(tp1_));
  }
}
