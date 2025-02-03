#ifndef _TENSOR__SCALAPACK_HPP
#define _TENSOR__SCALAPACK_HPP

#include "core/typedefs.hpp"

namespace qtnh {
  extern "C" void blacs_get_(int*, int*, int*);
  extern "C" void blacs_pinfo_(int*, int*);
  extern "C" void blacs_gridinit_(int*, char*, int*, int*);
  extern "C" void blacs_gridinfo_(int*, int*, int*, int*, int*);
  extern "C" void descinit_(int*, int*, int*, int*, int*, int*, int*, int*, int*, int*);
  extern "C" void pdpotrf_(char*, int*, double*, int*, int*, int*, int*);
  extern "C" void blacs_gridexit_(int*);
  extern "C" int numroc_(int*, int*, int*, int*, int*);

  extern "C" void pdgesvd_(char* jobu, char* jobvt, int* m, int* n, 
                          double* a, int* ia, int* ja, int* desc_a, 
                          double* s, double* u, int* iu, int* ju, int* desc_u, 
                          double* vt, int* ivt, int* jvt, int* desc_vt, 
                          double* work, int* lwork, int* info);
  extern "C" void pzgesvd_(char* jobu, char* jobvt, int* m, int* n, 
                          qtnh::tel* a, int* ia, int* ja, int* desc_a, 
                          qtnh::tel* s, qtnh::tel* u, int* iu, int* ju, int* desc_u, 
                          qtnh::tel* vt, int* ivt, int* jvt, int* desc_vt, 
                          qtnh::tel* work, int* lwork, double* rwork, int* info);
}

#endif