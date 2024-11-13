#include <algorithm>
#include <iostream>
#include <numeric>

#include "con/pair-defs.hpp"
#include "core/utils.hpp"
#include "tensor/indexing.hpp"


namespace qtnh {
  template<> qtnh::tptr PairContractor<DenseTensor, DenseTensor>::contract() {
    return Tensor::cast<Tensor>(std::move(tp1_));
  }
}
