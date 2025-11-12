#ifndef QTNH_BLAS_WRAPPERS_HPP_INCLUDE
#define QTNH_BLAS_WRAPPERS_HPP_INCLUDE

#include <tuple>

#include "blas/matrix.hpp"

namespace qtnh {
  namespace lalg {
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& matrix);
    std::tuple<BlockCyclicMatrix, BlockCyclicMatrix> PZGEQRD(BlockCyclicMatrix&& matrix);
    std::tuple<BlockCyclicMatrix, BlockCyclicMatrix> PZGELQD(BlockCyclicMatrix&& matrix);
    std::tuple<BlockCyclicMatrix, BlockCyclicMatrix, BlockCyclicMatrix> PZGEQPD(BlockCyclicMatrix&& matrix);

    BlockCyclicMatrix PZGEMM(BlockCyclicMatrix&& a, BlockCyclicMatrix&& b, 
                             bool use_at = false, bool use_bt = false);
    
    BlockCyclicMatrix PZGEADD(BlockCyclicMatrix&& a, BlockCyclicMatrix&& c, 
                              tel alpha, tel beta, bool use_at = false);
    
    BlockCyclicMatrix PZLAPIV(BlockCyclicMatrix&& matrix, const BlockCyclicMatrix& pv, pvec ipiv);
    BlockCyclicMatrix PZLAPV2(BlockCyclicMatrix&& matrix, 
                              const BlockCyclicMatrix& pv, pvec ipiv, 
                              char direc, char rowcol);
  
    void PZLAPRNT(const BlockCyclicMatrix& m, std::string id);
  }
}

#endif