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

MPO swapGate(const QTNHEnv& env, std::size_t n) {
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

void qft(const QTNHEnv& env, BCMPS& mps) {
  auto n = mps.nSites();

  for (auto i = 0UL; i < n; ++i) {
    if (utils::is_root()) {
      std::cout << "Iteration " << i + 1 << "/" << n << "\n";
    }

    mps.apply(qops::h(env), { i });
    mps.leftCanonicalise(i);
    mps.rightCanonicalise(i);

    if (i + 1 == n) break;

    MPO mpo = qops::cmp(env, n - i);
    mpo.rightCanonicalise();
    mps.apply(mpo, i);
  }
}

int main() {
  QTNHEnv env;

  constexpr auto N_SITES  = 4UL;
  constexpr auto SITE_DIM = 2UL;
  constexpr auto CHI_CYC  = 3UL;
  constexpr auto CHI_DIS  = 2UL;
  constexpr auto CHI_BLK  = 4UL;

  BCMPS mps(env, N_SITES, SITE_DIM, { CHI_CYC, CHI_DIS, CHI_BLK });

  mps.print();

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

  mps.print();
  mps.rightCanonicalise(0);

  mps.apply(cswap34, 0);
  mps.rightCanonicalise(0);

  mps.leftCanonicalise(3);
  mps.rightCanonicalise(0);
  mps.print();

  tptr tp = std::move(mps).toDense();
  tp->print_serial("Statevec");

  BCMPS mps2(env, N_SITES, SITE_DIM, { CHI_CYC, CHI_DIS, CHI_BLK });
  qft(env, mps2);

  auto norm = mps2.norm();
  if (utils::is_root()) {
    std::cout << "|T| = " << norm << "\n";
  }

  mps2.print();
  tp = std::move(mps2).toDense();
  tp->print_serial("QFT-SVec");

  return 0;
}
