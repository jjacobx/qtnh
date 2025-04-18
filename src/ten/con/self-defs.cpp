#include <algorithm>
#include <iostream>
#include <numeric>

#include "ten/con/self-defs.hpp"
#include "util/utils.hpp"
#include "util/indexing.hpp"

namespace qtnh {
  template<> qtnh::tptr SelfContractor<DenseTensor>::contract() {
    return Tensor::cast<Tensor>(std::move(tp1_));
  }
}
