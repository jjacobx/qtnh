#ifndef QTNH_BLAS_ROUTINES_HPP_INCLUDE
#define QTNH_BLAS_ROUTINES_HPP_INCLUDE

#include <complex>

namespace qtnh {
  namespace lalg {
    using complex = std::complex<double>;

    // Cblacs. 
    extern "C" void Cblacs_pinfo(int* mypnum, int* nprocs);
    extern "C" void Cblacs_get(int context, int request, int* value);
    extern "C" void Cblacs_exit(int error_code);
    extern "C" void Cblacs_barrier(int context, char* scope);

    extern "C" void Cblacs_gridinit(int* context, char* order, int np_row, int np_col);
    extern "C" void Cblacs_gridmap(int* context, int* usermap, int ldup, int np_row, int np_col);
    extern "C" void Cblacs_gridinfo(int context, int* np_row, int* np_col, int* my_row, int* my_col);
    extern "C" void Cblacs_gridexit(int context);

    extern "C" int  Cblacs_pnum(int context, int prow, int pcol);
    extern "C" void Cblacs_pcoord(int context, int pnum, int* prow, int* pcol);

    // ScaLAPACK. 
    extern "C" void pzlacpy_(char* uplo, int* m, int* n, 
                             complex* a, int* ia, int* ja, int* desc_a, 
                             complex* b, int* ib, int* jb, int* desc_b);
    extern "C" void pzlaset_(char* uplo, int* m, int* n, complex* alpha, complex* beta, 
                             complex* a, int* ia, int* ja, int* desc_a);

    extern "C" void pzgesvd_(char* jobu, char* jobvt, int* m, int* n, 
                             complex* a, int* ia, int* ja, int* desc_a, double* s, 
                             complex* u, int* iu, int* ju, int* desc_u, 
                             complex* vt, int* ivt, int* jvt, int* desc_vt, 
                             complex* work, int* lwork, double* rwork, int* info);
    
    extern "C" void pzgemm_(char* transa, char* transb, int* m, int* n, int* k, complex* alpha, 
                            complex* a, int* ia, int* ja, int* desc_a, 
                            complex* b, int* ib, int* jb, int* desc_b, complex* beta, 
                            complex* c, int* ic, int* jc, int* desc_c);
    
    extern "C" void pzgeqrf_(int* m, int* n, 
                             complex* a, int* ia, int* ja, int* desc_a, 
                             complex* tau, complex* work, int* lwork, int* info);
    extern "C" void pzungqr_(int* m, int* n, int* k, 
                             complex* a, int* ia, int* ja, int* desc_a, 
                             complex* tau, complex* work, int* lwork, int* info);
    
    extern "C" void pzgelqf_(int* m, int* n, 
                             complex* a, int* ia, int* ja, int* desc_a, 
                             complex* tau, complex* work, int* lwork, int* info);
    extern "C" void pzunglq_(int* m, int* n, int* k, 
                             complex* a, int* ia, int* ja, int* desc_a, 
                             complex* tau, complex* work, int* lwork, int* info);
    
  }
}

#endif