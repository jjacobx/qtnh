#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::complex_literals;

std::unique_ptr<SymmTensor> H(const QTNHEnv& env) {
  std::vector<tel> els_h = {
    1 / std::sqrt(2),  1 / std::sqrt(2), 
    1 / std::sqrt(2), -1 / std::sqrt(2)
  };

  return SymmTensor::make(env, {}, { 2, 2 }, std::move(els_h));
}

MPO controlledGate(const QTNHEnv& env, std::size_t n, std::vector<tel> els) {
  std::vector<tel> c_op { 1, 0, 0, 0, 0, 0, 0, 1 };
  std::vector<tel> i_op { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1 };
  std::vector<tel> t_op { 1, 0, 0, 1, els.at(0), els.at(1), els.at(2), els.at(3) };

  std::vector<tptr> ops(n);

  ops.at(0) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(c_op));
  ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });

  for (auto i = 1UL; i < n - 1; ++i) {
    ops.at(i) = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, std::vector<tel>(i_op));
    ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
  }

  ops.at(n - 1) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(t_op));
  ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });

  return MPO(std::move(ops));
}

MPO swapGate(const QTNHEnv& env, std::size_t n, std::vector<tel>) {
  std::vector<tel> l_op {
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
  };
  std::vector<tel> i_op { 
    1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1
  };
  std::vector<tel> r_op {
    1, 0, 0, 0,
    0, 0, 1, 0,
    0, 1, 0, 0,
    0, 0, 0, 1
  };

  std::vector<tptr> ops(n);

  ops.at(0) = DenseTensor::make(env, {}, { 4, 2, 2 }, std::move(l_op));
  ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });

  for (auto i = 1UL; i < n - 1; ++i) {
    ops.at(i) = DenseTensor::make(env, {}, { 4, 4, 2, 2 }, std::vector<tel>(i_op));
    ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
  }

  ops.at(n - 1) = DenseTensor::make(env, {}, { 4, 2, 2 }, std::move(r_op));
  ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });

  return MPO(std::move(ops));
}

void qft(const QTNHEnv& env, MPS& mps) {
  auto n = mps.nSites();

  for (auto i = 0UL; i < n; ++i) {
    mps.apply(H(env), { i });
    for (auto j = i + 1; j < n; ++j) {
      auto k = j - i + 1;
      auto c = std::exp(2i * M_PI / std::pow(2, k));
      MPO cp_mpo = controlledGate(env, k, { 1, 0, 0, c });
      mps.apply(cp_mpo, i);
    }
  }
}

int main() {
  QTNHEnv env;

  constexpr auto N_SITES  = 4UL;
  constexpr auto SITE_DIM = 2UL;
  constexpr auto CHI_CYC  = 2UL;
  constexpr auto CHI_DIS  = 2UL;
  constexpr auto CHI_BLK  = 2UL;

  BCMPS mps(env, N_SITES, SITE_DIM, { CHI_CYC, CHI_DIS, CHI_BLK });

  // mps.print();

  std::vector<tel> els_x = {
    0, 1, 
    1, 0
  };

  std::vector<tel> els_h = {
    1 / std::sqrt(2),  1 / std::sqrt(2), 
    1 / std::sqrt(2), -1 / std::sqrt(2)
  };

  std::vector<tel> els_hh = {
    .5,  .5,  .5,  .5, 
    .5, -.5,  .5, -.5, 
    .5,  .5, -.5, -.5, 
    .5, -.5, -.5,  .5
  };

  std::vector<tel> els_cx = {
    1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 0, 1, 
    0, 0, 1, 0
  };

  auto x =  SymmTensor::make(env, {}, tidx_tup(2, SITE_DIM), std::move(els_x));
  auto h =  SymmTensor::make(env, {}, tidx_tup(2, SITE_DIM), std::move(els_h));
  auto hh = SymmTensor::make(env, {}, tidx_tup(4, SITE_DIM), std::move(els_hh));
  auto cx = SymmTensor::make(env, {}, tidx_tup(4, SITE_DIM), std::move(els_cx));

  utils::barrier();

  mps.apply(Tensor::cast<SymmTensor>(h->copy()), { 0 });
  mps.apply(Tensor::cast<SymmTensor>(x->copy()), { 3 });
  // mps.apply(Tensor::cast<SymmTensor>(h->copy()), { 2 });
  // mps.apply(Tensor::cast<SymmTensor>(h->copy()), { 3 });
  // mps.print();

  // MPO cp_mpo = controlledGate(env, 3, { 1, 0, 0, 0.0 + 1.0i });
  // cp_mpo.print();
  // mps.apply(cp_mpo, 0);

  auto swap2 = qops::swap(env, 2);
  auto swap3 = qops::swap(env, 3);

  auto cswap24 = qops::cmpo(env, swap2, 4);
  auto cswap34 = qops::cmpo(env, swap3, 4);
  cswap24.rightCanonicalise();
  cswap34.rightCanonicalise();

  mps.apply(cswap24, 0);
  mps.rightCanonicalise(0);

  mps.apply(cswap34, 0);
  mps.rightCanonicalise(0);

  // mps.leftCanonicalise(3);
  // mps.rightCanonicalise(0);
  // mps.print();

  // qft(env, mps);
  auto norm = mps.norm();
  if (utils::is_root()) std::cout << norm << "\n";
  // mps.renormalise();

  // mps.print();

  BCMPS zero_amp(env, N_SITES, SITE_DIM, { 1, 1, 1 });
  auto overlap = mps.overlap(zero_amp);
  if (utils::is_root()) std::cout << "T[0] = " << overlap << "\n";

  auto swap_targets = qops::rotate_swaps(8, 4);
  if (utils::is_root()) std::cout << swap_targets << "\n";
  if (utils::is_root()) std::cout << mps.bondDims() << "\n";

  auto tp = std::move(mps).toDense();
  tp->print_serial("Psi");

  // auto swap = qops::swap(env, 3);
  // auto cswap = qops::cmpo(env, swap, 4);
  // cswap.at(3).print_serial("Last");

  // cswap.rightCanonicalise();
  // cswap.print();

  // auto tp = cswap.extract(3);

  // tp->print_serial("TP");

  // tp = Tensor::permute(std::move(tp), { 1, 2, 0 });

  // DecParams dp {{ 0, 0 }, { 2, 1 }, { 2, 1 }, { 0, 0 }, { 0, 0 }};
  // Decomposer dec(std::move(tp), dp, true);
  // dec.decompose();

  // auto [tp_u, tp_s, tp_v] = dec.extract_results();

  // tp_u->print_serial("U");
  // tp_s->print_serial("S");
  // tp_v->print_serial("V");

  tptr tp_site = DenseTensor::make(env, {}, { 2, 2, 2, 4, 2, 4 }, std::vector<tel> {
    // -1.71907, -0.211628, 0, 0, 0.992508, 0.122183, 0, 0, 
    // 0.211628, -1.71907, 0, 0, -0.122183, 0.992508, 0, 0, 
    // 0, 0, 1.73205, 0, 0, 0, -1, 0, 
    // 0, 0, 0, 1.73205, 0, 0, 0, -1, 
  
    // -1.71907, -0.211628, 0, 0, -0.992508, -0.122183, 0, 0, 
    // 0.211628, -1.71907, 0, 0, 0.122183, -0.992508, 0, 0, 
    // 0, 0, 1.73205, 0, 0, 0, 1, 0, 
    // 0, 0, 0, 1.73205, 0, 0, 0, 1,
  
  
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
  
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0,
  
  
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
  
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0, 
    // 0, 0, 0, 0, 0, 0, 0, 0,
  
  
    // 1.73205, 0, 0, 0, -1, 0, 0, 0, 
    // 0, 1.73205, 0, 0, 0, -1, 0, 0, 
    // 0, 0, 1.73205, 0, 0, 0, -1, 0, 
    // 0, 0, 0, 1.73205, 0, 0, 0, -1, 
  
    // 1.73205, 0, 0, 0, 1, 0, 0, 0, 
    // 0, 1.73205, 0, 0, 0, 1, 0, 0, 
    // 0, 0, 1.73205, 0, 0, 0, 1, 0, 
    // 0, 0, 0, 1.73205, 0, 0, 0, 1

    0.41, 0.72, 0.  , 0.3 , 0.14, 0.09, 0.18, 0.34, 0.39, 0.53, 0.41,
    0.68, 0.2 , 0.87, 0.02, 0.67, 0.41, 0.55, 0.14, 0.19, 0.8 , 0.96,
    0.31, 0.69, 0.87, 0.89, 0.08, 0.03, 0.16, 0.87, 0.09, 0.42, 0.95,
    0.53, 0.69, 0.31, 0.68, 0.83, 0.01, 0.75, 0.98, 0.74, 0.28, 0.78,
    0.1 , 0.44, 0.9 , 0.29, 0.28, 0.13, 0.01, 0.67, 0.21, 0.26, 0.49,
    0.05, 0.57, 0.14, 0.58, 0.69, 0.1 , 0.41, 0.69, 0.41, 0.04, 0.53,
    0.66, 0.51, 0.94, 0.58, 0.9 , 0.13, 0.13, 0.8 , 0.39, 0.16, 0.92,
    0.34, 0.75, 0.72, 0.88, 0.62, 0.75, 0.34, 0.26, 0.89, 0.42, 0.96,
    0.66, 0.62, 0.11, 0.94, 0.44, 0.57, 0.4 , 0.23, 0.9 , 0.57, 0.  ,
    0.61, 0.32, 0.52, 0.88, 0.35, 0.9 , 0.62, 0.01, 0.92, 0.69, 0.99,
    0.17, 0.13, 0.93, 0.69, 0.06, 0.75, 0.75, 0.92, 0.71, 0.12, 0.01,
    0.02, 0.02, 0.24, 0.86, 0.53, 0.55, 0.84, 0.12, 0.27, 0.58, 0.96,
    0.56, 0.01, 0.8 , 0.23, 0.8 , 0.38, 0.86, 0.74, 0.55, 0.13, 0.05,
    0.12, 0.04, 0.1 , 0.22, 0.71, 0.55, 0.01, 0.07, 0.96, 0.56, 0.2 ,
    0.25, 0.74, 0.19, 0.58, 0.97, 0.84, 0.23, 0.49, 0.61, 0.82, 0.15,
    0.01, 0.07, 0.48, 0.6 , 0.56, 0.31, 0.98, 0.57, 0.38, 0.55, 0.74,
    0.66, 0.26, 0.06, 0.37, 0.62, 0.21, 0.75, 0.06, 0.26, 0.8 , 0.19,
    0.63, 0.52, 0.92, 0.26, 0.06, 0.73, 0.77, 0.9 , 0.93, 0.01, 0.23,
    0.61, 0.94, 0.95, 0.55, 0.91, 0.64, 0.39, 0.48, 0.6 , 0.54, 0.92,
    0.91, 0.39, 0.96, 0.17, 0.12, 0.13, 0.5 , 0.02, 0.94, 0.82, 0.01,
    0.17, 0.33, 0.13, 0.8 , 0.34, 0.94, 0.58, 0.87, 0.84, 0.9 , 0.45,
    0.54, 0.79, 0.28, 0.49, 0.59, 0.01, 0.59, 0.43, 0.8 , 0.31, 0.89,
    0.57, 0.18, 0.78, 0.61, 0.05, 0.42, 0.67, 0.91, 0.  , 0.97, 0.37,
    0.97, 0.6 , 0.82
  });

  tp_site = Tensor::rescatter(std::move(tp_site), 2);
  tp_site = Tensor::permute(std::move(tp_site), { 0, 1, 4, 5, 2, 3 });

  // DecParams dp {
  //   { 1, 1 }, 
  //   { 2, 2 }, 
  //   { 1, 1 }, 
  //   { 1, 1 }, 
  //   { 1, 1 }
  // };

  // Decomposer dec(std::move(tp_site), dp, true);
  // dec.decompose();

  // auto [tp_u, tp_s, tp_v] = dec.extract_results();

  lalg::ProcGrid pg(2, 2);
  lalg::BlockCyclicMatrix m(pg, { 4, 4 }, { 2, 2 }, { 2, 2 }, 
                            tp_site->cast<DenseTensor>()->extractEls());

  auto [u, s, v] = lalg::PZGESVD(std::move(m));

  if (utils::is_root()) std::cout << s << "\n";

  tptr tp_u = DenseTensor::make(env, { 2, 2 }, { 2, 4, 2, 4 }, u.extractEls());
  tptr tp_v = DenseTensor::make(env, { 2, 2 }, { 2, 4, 2, 4 }, v.extractEls());

  tp_u = Tensor::permute(std::move(tp_u), { 0, 1, 4, 5, 2, 3 });
  tp_v = Tensor::permute(std::move(tp_v), { 0, 1, 4, 5, 2, 3 });

  tptr tp_s = DiagTensor::make(env, {}, { 2, 2, 4, 2, 2, 4 }, false, std::move(s));
  tp_s = SymmTensorBase::rescatterIO(std::move(tp_s), 1);

  pcon con1(std::move(tp_u), std::move(tp_s), ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }}));
  tptr tp_us = con1.contract();

  pcon con2(std::move(tp_us), std::move(tp_v), ConParams({{ 1, 0 }, { 4, 2 }, { 5, 3 }}));
  tptr tp_usv = con2.contract();

  tp_usv->print_serial("USV");

  return 0;
}
