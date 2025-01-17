#include <algorithm>
#include <iostream>
#include <numeric>

#include "con/self-defs.hpp"
#include "core/utils.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  template<> qtnh::tptr SelfContractor<DenseTensor>::contract() {
    return Tensor::cast<Tensor>(std::move(tp1_));
  }
}
