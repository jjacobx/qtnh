#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  int nprow = 1;   // Number of row procs
  int npcol = 1;   // Number of column procs
  char layout = 'R'; // Block cyclic, Row major processor mapping

  int iam, nprocs;
  int zero = 0;
  int ictxt, myrow, mycol;
  blacs_pinfo_(&iam, &nprocs) ; // BLACS rank and world size
  blacs_get_(&zero, &zero, &ictxt ); // -> Create context
  blacs_gridinit_(&ictxt, &layout, &nprow, &npcol ); // Context -> Initialize the grid
  blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol );

  std::cout << iam << ", " << nprocs << "," << ictxt << "\n";

  char jobu = 'V', jobvt = 'V';
  int m = 2, n = 2;
  double a_d[] = { 1, 0, 0, 1 }; tel a_c[] = { 1, 0, 0, 1 };
  int ia = 1, ja = 1;
  int desc_a[] = { 1, ictxt, 2, 2, 1, 1, 0, 0, 2 };

  double s_d[4]; tel s_c[4];
  double u_d[4]; tel u_c[4];
  int iu = 1, ju = 1;
  int desc_u[] = { 1, ictxt, 2, 2, 1, 1, 0, 0, 2 };

  double vt_d[4]; tel vt_c[4];
  int ivt = 1, jvt = 1;
  int desc_vt[] = { 1, ictxt, 2, 2, 1, 1, 0, 0, 2 };

  double work_d[10000]; tel work_c[10000];
  int lwork = 10000;
  double rwork[10000];
  int lrwork = 10000;
  int info = 0;

  //pdgesvd_(&jobu, &jobvt, &m, &n, a_d, &ia, &ja, desc_a, s_d, u_d, &iu, &ju, desc_u, vt_d, &ivt, &jvt, desc_vt, work_d, &lwork, &info);
  pzgesvd_(&jobu, &jobvt, &m, &n, a_c, &ia, &ja, desc_a, s_c, u_c, &iu, &ju, desc_u, vt_c, &ivt, &jvt, desc_vt, work_c, &lwork, rwork, &info);

  std::cout << info << "\n";
  std::cout << u_c[0] << ", " << u_c[1] << ", " << u_c[2] << ", " << u_c[3] << "\n";
  std::cout << vt_c[0] << ", " << vt_c[1] << ", " << vt_c[2] << ", " << vt_c[3] << "\n";

  return 0;
}
