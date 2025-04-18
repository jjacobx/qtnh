#ifndef __BLAS_WRAPPERS__
#define __BLAS_WRAPPERS__

#include "blas/matrix.hpp"
#include "util/typedefs.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& matrix);
    BlockCyclicMatrix PZGEMM(BlockCyclicMatrix&& a, BlockCyclicMatrix&& b, 
                             bool use_at = false, bool use_bt = false);
  }
}

#endif