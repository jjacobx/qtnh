#ifndef __LALG_WRAPPERS__
#define __LALG_WRAPPERS__

#include "core/typedefs.hpp"
#include "lalg/matrix.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& matrix);
    BlockCyclicMatrix PZGEMM(BlockCyclicMatrix&& a, BlockCyclicMatrix&& b, bool use_bt = false);
  }
}

#endif