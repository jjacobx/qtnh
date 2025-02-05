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

  if (grid.context != -1) decomp::blacs_gridexit_(&grid.context);
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

  // 1. Create block matrix structure
  // (ijk|l,mno) -> (ij,m)|(kl,no): (0 1 4 2 3 5 6)
  // 2. Switch to column-major form
  // (ij,m)|(kl,no) -> (ij,m)|(no,kl): (0 1 4 5 6 2 3)

  std::vector<qtnh::tidx_tup_st> ptup { 0, 1, 4, 5, 6, 2, 3 };
  tp = Tensor::permute(std::move(tp), ptup);
  auto dtp = Tensor::convert<DenseTensor>(std::move(tp));

  using namespace ops;
  std::cout << "P" << env.proc_id << ": " << *dtp << "\n";
  utils::barrier();
  
  using namespace lalg;
  ProcGrid pg42(4, 2);

  BlockMatrix bm(pg42, 16, 8, dtp->extractEls());
  auto desc = bm.descriptor();

  std::cout << "P" << env.proc_id << ": ";
  for (auto i = 0UL; i < 9; ++i) {
    std::cout << desc.at(i) << ", ";
  }
  std::cout << bm.grid().active() << "\n";

  auto mdims = bm.totDims();
  std::vector<tel> svals(8);

  BlockMatrix bm_u(pg42, 16, 8);
  BlockMatrix bm_v(pg42, 8, 8);

  desc = bm_v.descriptor();
  std::cout << "P" << env.proc_id << ": ";
  for (auto i = 0UL; i < 9; ++i) {
    std::cout << desc.at(i) << ", ";
  }
  std::cout << bm.grid().active() << "\n";

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

  tptr tp_u = DenseTensor::make(env, { 2, 2, 2 }, { 2, 2, 2, 2 }, std::move(bm_u.extractEls()));
  std::vector<qtnh::tidx_tup_st> ptup_u { 0, 1, 5, 6, 2, 3, 4 };
  tp_u = Tensor::permute(std::move(tp_u), ptup_u);
  std::cout << "P" << env.proc_id << ": U = " << *tp_u << "\n";
  utils::barrier();

  tptr tp_v = DenseTensor::make(env, { 2, 2, 2 }, { 2, 2, 2 }, std::move(bm_v.extractEls()));
  std::vector<qtnh::tidx_tup_st> ptup_v { 0, 2, 1, 3, 4, 5 };
  tp_v = Tensor::permute(std::move(tp_v), ptup_v);
  tp_v = Tensor::rescatter(std::move(tp_v), -1);

  ptup_v = { 0, 3, 1, 4, 5, 2 };
  tp_v = Tensor::permute(std::move(tp_v), ptup_v);
  std::cout << "P" << env.proc_id << ": V = " << *tp_v << "\n";
  utils::barrier();

  tptr tp_s = DiagTensor::make(env, {}, { 2, 2, 2, 2, 2, 2 }, false, std::move(svals));
  tp_s = Tensor::convert<DenseTensor>(std::move(tp_s));
  tp_s = Tensor::permute(std::move(tp_s), { 0, 2, 3, 1, 4, 5 });
  tp_s = Tensor::rescatter(std::move(tp_s), 2);
  std::cout << "P" << env.proc_id << ": S = " << *tp_s << "\n";
  utils::barrier();

  auto params = ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }});
  auto con = pcon(std::move(tp_s), std::move(tp_v), params);
  auto tp_sv = con.contract();

  std::cout << "P" << env.proc_id << ": SV = " << *tp_sv << "\n";
  utils::barrier();

  params = ConParams({{ 2, 0 }, { 5, 2 }, { 6, 3 }});
  con = pcon(std::move(tp_u), std::move(tp_sv), params);
  auto tp_usv = con.contract();

  std::cout << "P" << env.proc_id << ": USV = " << *tp_usv << "\n";
  utils::barrier();

  return 0;
}
