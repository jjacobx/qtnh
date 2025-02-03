#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  int iam, nprocs;
  decomp::blacs_pinfo_(&iam, &nprocs); // BLACS rank and world size

  tidx_tup dims { 2, 2, 2, 2, 2, 2, 2, 2 };
  std::vector<tel> els(utils::dims_to_size(dims), 0);
  tptr tp = DenseTensor::make(env, {}, dims, std::move(els));

  tp = Tensor::rescatter(std::move(tp), 3);
  auto grid = decomp::tensor_to_grid(tp.get(), 2);

  std::cout << "1. Process: " << iam << "/" << nprocs << "; Context: " << grid.context << 
    "; Coordinates = (" << grid.row << ", " << grid.col << ")\n";

  decomp::blacs_gridexit_(&grid.context);
  utils::barrier();

  int nprow = 2;      // Number of row procs
  int npcol = 3;      // Number of column procs
  char layout = 'R';  // Block cyclic, Row major processor mapping
  
  int zero = 0;
  int ictxt, myrow, mycol;
  decomp::blacs_get_(&zero, &zero, &ictxt); // -> Create context
  decomp::blacs_gridinit_(&ictxt, &layout, &nprow, &npcol); // Context -> Initialize the grid
  decomp::blacs_gridinfo_(&ictxt, &nprow, &npcol, &myrow, &mycol);

  std::cout << "2. Process: " << iam << "/" << nprocs << "; Context: " << ictxt << 
    "; Coordinates = (" << myrow << ", " << mycol << ")\n";

  char jobu = 'V', jobvt = 'V';
  int m = 2, n = 2;
  tel a_c[] = { 1, 0, 0, 1 };
  int ia = 1, ja = 1;
  int desc_a[] = { 1, ictxt, 2, 2, 1, 1, 0, 0, 2 };

  tel s_c[4];
  tel u_c[4];
  int iu = 1, ju = 1;
  int desc_u[] = { 1, ictxt, 2, 2, 1, 1, 0, 0, 2 };

  tel vt_c[4];
  int ivt = 1, jvt = 1;
  int desc_vt[] = { 1, ictxt, 2, 2, 1, 1, 0, 0, 2 };

  tel work_c[10000];
  int lwork = 10000;
  double rwork[10000];
  int info = 0;

  decomp::pzgesvd_(&jobu, &jobvt, &m, &n, a_c, &ia, &ja, desc_a, s_c, 
                   u_c, &iu, &ju, desc_u, vt_c, &ivt, &jvt, desc_vt, 
                   work_c, &lwork, rwork, &info);

  std::cout << info << "\n";
  std::cout << u_c[0] << ", " << u_c[1] << ", " << u_c[2] << ", " << u_c[3] << "\n";
  std::cout << vt_c[0] << ", " << vt_c[1] << ", " << vt_c[2] << ", " << vt_c[3] << "\n";

  return 0;
}
