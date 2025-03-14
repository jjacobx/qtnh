#ifndef __BLAS_WRAPPERS__
#define __BLAS_WRAPPERS__

#include "ten/util/typedefs.hpp"
#include "blas/matrix.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& matrix);
    BlockCyclicMatrix PZGEMM(BlockCyclicMatrix&& a, BlockCyclicMatrix&& b, 
                             bool use_at = false, bool use_bt = false);
  }
}

#endif