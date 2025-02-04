#include <iostream>
#include <numeric>
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

  if (ictxt != -1) decomp::blacs_gridexit_(&ictxt);

  tidx_tup dims2 { 2, 2, 2, 2, 2, 2, 2 };
  els = std::vector<tel>(utils::dims_to_size(dims2), 0);
  std::iota(els.begin(), els.end(), 0);
  tp = DenseTensor::make(env, {}, dims2, std::move(els));
  tp = Tensor::rescatter(std::move(tp), 3);

  // Create block matrix structure
  // (2, 2, 2) (2, 2, 2, 2) -> (2, 2; 2) (2, 2; 2, 2)
  // 2 -> 4; 0 1 3 4 2 5 6

  std::vector<tidx_tup_st> ptup1(7, 0);
  std::iota(ptup1.begin(), ptup1.end(), 0);
  auto pel = ptup1.at(2);
  ptup1.erase(ptup1.begin() + 2);
  ptup1.insert(ptup1.begin() + 4, pel);

  tp = Tensor::permute(std::move(tp), ptup1);
  
  // Switch to column-major form
  // 0 1 3 4 2 5 6 -> 0 1 3 5 6 4 2

  std::vector<tidx_tup_st> ptup2(7, 0);
  std::iota(ptup2.begin(), ptup2.end(), 0);
  auto pels = std::vector<tidx_tup_st>(ptup2.begin() + 3, ptup2.begin() + 4);
  ptup2.erase(ptup2.begin() + 3, ptup2.begin() + 4);
  ptup2.insert(ptup2.end(), pels.begin(), pels.end());

  tp = Tensor::permute(std::move(tp), ptup2);
  auto dtp = Tensor::convert<DenseTensor>(std::move(tp));
  auto mels = dtp->locElsP();

  using namespace ops;
  std::cout << "P" << env.proc_id << ": " << *dtp << "\n";
  utils::barrier();
  
  using namespace lalg;
  ProcGrid pgrid(4, 2);

  BlockMatrix bm(pgrid, 16, 8, mels);
  auto desc = bm.descriptor();

  std::cout << "P" << env.proc_id << ": ";
  for (auto i = 0UL; i < 9; ++i) {
    std::cout << desc.at(i) << ", ";
  }
  std::cout << bm.grid().active() << "\n";

  auto mdims = bm.totDims();
  std::vector<tel> svals(8);

  BlockMatrix bm_u(pgrid, 16, 8);
  BlockMatrix bm_v(pgrid, 8, 8);

  auto one = 1;

  tel work2[10000];
  int lwork2 = 10000;
  double rwork2[10000];
  auto info2 = -1;

  if (bm.grid().active()) {
    // Won't be modified, so must const cast. 
    auto a_desc = bm.descriptor();
    auto a_desc_d = const_cast<int*>(a_desc.data());
    auto u_desc = bm_u.descriptor();
    auto u_desc_d = const_cast<int*>(u_desc.data());
    auto v_desc = bm_v.descriptor();
    auto v_desc_d = const_cast<int*>(v_desc.data());
    pzgesvd_(&jobu, &jobvt, &mdims.first, &mdims.second, bm.data(), &one, &one, a_desc_d, svals.data(), 
            bm_u.data(), &one, &one, u_desc_d, bm_v.data(), &one, &one, v_desc_d, 
            work2, &lwork2, rwork2, &info2);
  }

  // Needed to prevent double-freeing. 
  bm.extractEls().release();

  return 0;
}
