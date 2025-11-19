#ifndef QTNH_BLAS_ROUTINES_HPP_INCLUDE
#define QTNH_BLAS_ROUTINES_HPP_INCLUDE

#include <complex>

namespace qtnh {
  namespace lalg {
    using complex = std::complex<double>;

    extern "C" {
      void Cblacs_pinfo(int* mypnum, int* nprocs);
      void Cblacs_get(int context, int request, int* value);
      void Cblacs_exit(int error_code);
      void Cblacs_barrier(int context, char* scope);

      void Cblacs_gridinit(int* context, char* order, int np_row, int np_col);
      void Cblacs_gridmap(int* context, int* usermap, int ldup, int np_row, int np_col);
      void Cblacs_gridinfo(int context, int* np_row, int* np_col, int* my_row, int* my_col);
      void Cblacs_gridexit(int context);

      int  Cblacs_pnum(int context, int prow, int pcol);
      void Cblacs_pcoord(int context, int pnum, int* prow, int* pcol);

      // ScaLAPACK. 
      void pzlacpy_(char* uplo, int* m, int* n, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    complex* b, int* ib, int* jb, int* desc_b);
      void pzlaset_(char* uplo, int* m, int* n, complex* alpha, complex* beta, 
                    complex* a, int* ia, int* ja, int* desc_a);
    
      void pzgemm_(char* transa, char* transb, 
                   int* m, int* n, int* k, complex* alpha, 
                   complex* a, int* ia, int* ja, int* desc_a, 
                   complex* b, int* ib, int* jb, int* desc_b, complex* beta, 
                   complex* c, int* ic, int* jc, int* desc_c);

      void pzgeadd_(char* trans, int* m, int* n, 
                    complex* alpha, complex* a, int* ia, int*ja, int* desc_a, 
                    complex* beta, complex* c, int* ic, int* jc, int* desc_c);
    
      void pzgeqrf_(int* m, int* n, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    complex* tau, complex* work, int* lwork, int* info);
      void pzungqr_(int* m, int* n, int* k, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    complex* tau, complex* work, int* lwork, int* info);
    
      void pzgelqf_(int* m, int* n, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    complex* tau, complex* work, int* lwork, int* info);
      void pzunglq_(int* m, int* n, int* k, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    complex* tau, complex* work, int* lwork, int* info);

      void pzgeqpf_(int* m, int* n, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    int* ipiv, complex* tau, complex* work, int* lwork, 
                    double* rwork, int* lrwork, int* info);

      void pzgesvd_(char* jobu, char* jobvt, int* m, int* n, 
                    complex* a, int* ia, int* ja, int* desc_a, double* s, 
                    complex* u, int* iu, int* ju, int* desc_u, 
                    complex* vt, int* ivt, int* jvt, int* desc_vt, 
                    complex* work, int* lwork, double* rwork, int* info);
    
      void pzlapiv_(char* direc, char* rowcol, char* pivroc, int* m, int* n, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    int* ipiv, int* ip, int* jp, int* desc_ip, int* iwork);
      void pzlapv2_(char* direc, char* rowcol, int* m, int* n, 
                    complex* a, int* ia, int* ja, int* desc_a, 
                    int* ipiv, int* ip, int* jp, int* desc_ip);

      void pzlaprnt_(int* m, int* n, 
                     complex* a, int* ia, int* ja, int* desc_a, 
                     int* irprnt, int* icprnt, 
                     char* cmatnm, int* nout, complex* work, int size);
    }
  }
}

#endif