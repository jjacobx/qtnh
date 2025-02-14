#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  auto dims = tidx_tup { 2, 2, 2, 2, 2, 2, 2 };
  auto els = std::vector<tel>(utils::dims_to_size(dims), 0);
  std::iota(els.begin(), els.end(), 0);

  tptr tp = DenseTensor::make(env, {}, dims, std::move(els));

  auto ndis_dims = 3UL;
  tp = Tensor::rescatter(std::move(tp), ndis_dims);

  auto dis_sep = 2UL;
  auto loc_sep = 2UL;

  PTupleSrc ptup(dims.size());
  auto [tup_r, tup_c] = utils::split_vec(ptup.tup(), dis_sep + loc_sep);
  auto [tup_rd, tup_rb] = utils::split_vec(tup_r, dis_sep);
  auto [tup_cd, tup_cb] = utils::split_vec(tup_c, ndis_dims - dis_sep);

  IndexGroup ig({ "rd", "cd", "cb", "rb" }, { tup_rd, tup_cd, tup_cb, tup_rb });
  if (utils::is_root()) std::cout << ig.ptup().inv().tup() << "\n";

  tp = Tensor::permute(std::move(tp), ig.ptup().toTar().tup());
  auto dtp = Tensor::convert<DenseTensor>(std::move(tp));

  auto [dims_rd, dims_cd] = utils::split_dims(dtp->disDims(), dis_sep);
  auto [dims_cb, dims_rb] = utils::split_dims(dtp->locDims(), dtp->locDims().size() - loc_sep);
  auto size_rd = utils::dims_to_size(dims_rd), size_cd = utils::dims_to_size(dims_cd);
  auto size_cb = utils::dims_to_size(dims_cb), size_rb = utils::dims_to_size(dims_rb);
 
  using namespace lalg;
  ProcGrid pg(size_rd, size_cd);

  BlockMatrix bm(pg, size_rd * size_rb, size_cd * size_cb, dtp->extractEls());
  auto [u, s, v] = PZGESVD(std::move(bm));

  if (utils::is_root()) std::cout << "S = " << s << "\n";

  tptr tp_u = DenseTensor::make(env, { 2, 2, 2 }, { 2, 2, 2, 2 }, u.extractEls());
  tptr tp_v = DenseTensor::make(env, { 2, 2, 2 }, { 2, 2, 2 }, v.extractEls());
  tptr tp_s = DiagTensor::make(env, {}, { 2, 2, 2, 2, 2, 2 }, false, std::move(s));

  tp_u = Tensor::rescatter(std::move(tp_u), -1);
  tp_u = Tensor::permute(std::move(tp_u), ig.ptup().toTar().inv().tup());
  tp_v = Tensor::permute(std::move(tp_v), PTupleSrc({ 2, 0, 1, 5, 3, 4 }).toTar().tup());
  tp_v = Tensor::rescatter(std::move(tp_v), -2);
  tp_s = Tensor::convert<DenseTensor>(std::move(tp_s));

  auto params = ConParams({{ 3, 1 }, { 4, 2 }, { 5, 3 }});
  auto con = pcon(std::move(tp_s), std::move(tp_v), params);
  auto tp_sv = con.contract();

  params = ConParams({{ 4, 1 }, { 5, 2 }, { 6, 3 }});
  con = pcon(std::move(tp_u), std::move(tp_sv), params);
  auto tp_usv = con.contract();

  tp_usv = Tensor::permute(std::move(tp_usv), PTupleSrc({ 0, 1, 3, 4, 2, 5, 6 }).toTar().tup());

  std::cout << "P" << env.proc_id << ": USV = " << *tp_usv << "\n";
  utils::barrier();

  dims = tidx_tup { 2, 2, 2, 2, 2, 2, 2 };
  els = std::vector<tel>(utils::dims_to_size(dims), 0);
  std::iota(els.begin(), els.end(), 0);

  tp = DenseTensor::make(env, {}, dims, std::move(els));
  tp = Tensor::rescatter(std::move(tp), 2);

  DecParams dp {{ 1, 1 }, { 3, 2 }, { 2, 1 }, { 1, 1 }, { 1, 1 }};
  Decomposer dec(std::move(tp), dp);
  dec.decompose();

  auto [tu, ts, tv] = dec.extract_results();

  std::cout << "P" << env.proc_id << ": U = " << *tu << "\n";
  std::cout << "P" << env.proc_id << ": S = " << *ts << "\n";
  std::cout << "P" << env.proc_id << ": V = " << *tv << "\n";
  
  ts = DiagTensor::make(env, {}, utils::concat_dims(ts->totDims(), ts->totDims()), false, ts->cast<DenseTensor>()->extractEls());
  ts = Tensor::convert<DenseTensor>(std::move(ts));
  
  PTupleSrc s_ptup(ts->totDims().size());
  auto [tup1, tup_r1] = utils::split_dims(s_ptup.tup(), 1);
  auto [tup2, tup_r2] = utils::split_dims(tup_r1, 2);
  auto [tup3, tup4] = utils::split_dims(tup_r2, 1);
  IndexGroup igs({ "d1", "l1", "d2", "l2" }, { tup1, tup2, tup3, tup4 });
  igs.reorder({ "d1", "d2", "l1", "l2" });

  ts = Tensor::permute(std::move(ts), igs.ptup().toTar().tup());
  ts = Tensor::rescatter(std::move(ts), 2);

  params = ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }});
  con = pcon(std::move(ts), std::move(tv), params);
  auto tsv = con.contract();

  params = ConParams({{ 1, 0 }, { 5, 2 }, { 6, 3 }});
  con = pcon(std::move(tu), std::move(tsv), params);
  auto tusv = con.contract();
  tusv = Tensor::rebcast(std::move(tusv), { 1, 1, 0 });

  std::cout << "P" << env.proc_id << ": USV = " << *tusv << "\n";

  return 0;
}
