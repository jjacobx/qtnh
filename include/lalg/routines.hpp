#ifndef __LALG_ROUTINES__
#define __LALG_ROUTINES__

#include "core/typedefs.hpp"
#include "lalg/matrix.hpp"

namespace qtnh {
  namespace lalg {
    extern "C" void sl_init_(int*, int*, int*);
    extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
    extern "C" void blacs_gridexit_(int*);

    // Call this to make processes independent. 
    extern "C" void blacs_get_(int*, int*, int*);
    extern "C" void blacs_gridmap_(int*, int*, int*, int*, int*, int*);

    extern "C" void pzgesvd_(char* jobu, char* jobvt, int* m, int* n, 
                             qtnh::tel* a, int* ia, int* ja, int* desc_a, double* s, 
                             qtnh::tel* u, int* iu, int* ju, int* desc_u, 
                             qtnh::tel* vt, int* ivt, int* jvt, int* desc_vt, 
                             qtnh::tel* work, int* lwork, double* rwork, int* info);
    
    extern "C" void pzgemm_(char* transa, char* transb, int* m, int* n, int* k, qtnh::tel* alpha, 
                            qtnh::tel* a, int* ia, int* ja, int* desc_a, 
                            qtnh::tel* b, int* ib, int* jb, int* desc_b, qtnh::tel* beta, 
                            qtnh::tel* c, int* ic, int* jc, int* desc_c);
    
                             
    // Wrappers. 
    std::tuple<BlockCyclicMatrix, cvec, BlockCyclicMatrix> PZGESVD(BlockCyclicMatrix&& matrix);
  }
}

#endif