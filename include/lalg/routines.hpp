#ifndef __LALG_ROUTINES__
#define __LALG_ROUTINES__

#include <complex>

namespace qtnh {
  namespace lalg {
    using complex = std::complex<double>;

    extern "C" void sl_init_(int*, int*, int*);
    extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
    extern "C" void blacs_gridexit_(int*);

    extern "C" int  blacs_pnum_(int*, int*, int*);
    extern "C" void blacs_pcoord_(int*, int*, int*, int*);

    // Call this to make processes independent. 
    extern "C" void blacs_get_(int*, int*, int*);
    extern "C" void blacs_gridmap_(int*, int*, int*, int*, int*);

    extern "C" void pzgesvd_(char* jobu, char* jobvt, int* m, int* n, 
                             complex* a, int* ia, int* ja, int* desc_a, double* s, 
                             complex* u, int* iu, int* ju, int* desc_u, 
                             complex* vt, int* ivt, int* jvt, int* desc_vt, 
                             complex* work, int* lwork, double* rwork, int* info);
    
    extern "C" void pzgemm_(char* transa, char* transb, int* m, int* n, int* k, complex* alpha, 
                            complex* a, int* ia, int* ja, int* desc_a, 
                            complex* b, int* ib, int* jb, int* desc_b, complex* beta, 
                            complex* c, int* ic, int* jc, int* desc_c);
  }
}

#endif