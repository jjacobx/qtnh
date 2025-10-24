#ifndef QTNH_BLAS_WRAPPERS_HPP_INCLUDE
#define QTNH_BLAS_WRAPPERS_HPP_INCLUDE

#include <tuple>

#include "blas/matrix.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& matrix);
    BlockCyclicMatrix PZGEMM(BlockCyclicMatrix&& a, BlockCyclicMatrix&& b, 
                             bool use_at = false, bool use_bt = false);
  }
}

#endif