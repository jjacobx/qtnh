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

  tptr tp_m = DenseTensor::make(env, {}, dims, std::move(els));

  auto ndis_dims = 2UL;
  tp_m = Tensor::rescatter(std::move(tp_m), ndis_dims);

  DecParams dp {{ 1, 1 }, { 3, 2 }, { 2, 1 }, { 1, 1 }, { 1, 1 }};
  Decomposer dec(std::move(tp_m), dp);
  dec.decompose();

  auto [tp_u, tp_s, tp_v] = dec.extract_results();

  std::cout << "P" << env.proc_id << ": U = " << *tp_u << "\n";
  std::cout << "P" << env.proc_id << ": S = " << *tp_s << "\n";
  std::cout << "P" << env.proc_id << ": V = " << *tp_v << "\n";
  
  auto&& els_s = tp_s->cast<DenseTensor>()->extractEls();
  tp_s = DiagTensor::make(env, {}, utils::concat_dims(tp_s->totDims(), tp_s->totDims()), 0, std::move(els_s));
  tp_s = Tensor::convert<DenseTensor>(std::move(tp_s));
  
  PTupleSrc s_ptup(tp_s->totDims().size());
  auto [tup1, tup_r1] = utils::split_dims(s_ptup.tup(), 1);
  auto [tup2, tup_r2] = utils::split_dims(tup_r1, 2);
  auto [tup3, tup4] = utils::split_dims(tup_r2, 1);
  IndexGroup igs({ "d1", "l1", "d2", "l2" }, { tup1, tup2, tup3, tup4 });
  igs.reorder({ "d1", "d2", "l1", "l2" });

  tp_s = Tensor::permute(std::move(tp_s), igs.ptup().toTar().tup());
  tp_s = Tensor::rescatter(std::move(tp_s), 2);

  auto params = ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }});
  auto con = pcon(std::move(tp_s), std::move(tp_v), params);
  auto tp_sv = con.contract();

  params = ConParams({{ 1, 0 }, { 5, 2 }, { 6, 3 }});
  con = pcon(std::move(tp_u), std::move(tp_sv), params);
  auto tp_usv = con.contract();

  tp_usv = Tensor::rebcast(std::move(tp_usv), { 1, 1, 0 });

  std::cout << "P" << env.proc_id << ": USV = " << *tp_usv << "\n";

  dims = tidx_tup { 2, 4, 2, 4, 2 };
  els = std::vector<tel>(utils::dims_to_size(dims), 0);
  std::iota(els.begin(), els.end(), 0);

  tptr tp = DenseTensor::make(env, {}, dims, std::move(els));
  tp = Tensor::rescatter(std::move(tp), 2);

  tp = Tensor::truncate(std::move(tp), 1, 3);
  tp = Tensor::truncate(std::move(tp), 3, 1);
  tp->reshape(tp->disDims(), { 2, 2 });

  std::cout << "P" << env.proc_id << ": T = " << *tp << "\n";

  utils::barrier();
  if (utils::is_root()) std::cout << tp->totDims() << "\n";

  return 0;
}
