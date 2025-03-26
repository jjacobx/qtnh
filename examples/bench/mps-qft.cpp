#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
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

tel urot(int k) {
  return std::exp(2i * M_PI / std::pow(2, k));
}

MPO cMultiPhase(const QTNHEnv& env, std::size_t n) {
  std::vector<tptr> ops(n);
  std::vector<tel> op;

  op = { 1, 0, 0, 0, 0, 0, 0, 1 };
  ops.at(0) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(op));
  ops.at(0) = Tensor::permute(std::move(ops.at(0)), { 2, 1, 0 });

  for (auto i = 1UL; i + 1 < n; ++i) {
    op = { 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, urot(i + 1) };
    ops.at(i) = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, std::move(op));
    ops.at(i) = Tensor::permute(std::move(ops.at(i)), { 2, 3, 1, 0 });
  }
  
  op = { 1, 0, 0, 1, 1, 0, 0, urot(n) };
  ops.at(n - 1) = DenseTensor::make(env, {}, { 2, 2, 2 }, std::move(op));
  ops.at(n - 1) = Tensor::permute(std::move(ops.at(n - 1)), { 2, 1, 0 });

  return MPO(std::move(ops));
}

void qft(const QTNHEnv& env, MPS& mps) {
  auto n = mps.nSites();

  for (auto i = 0UL; i < n; ++i) {
    if (utils::is_root()) {
      std::cout << "Iteration " << i + 1 << "/" << n << "\n";
    }

    mps.apply(H(env), { i });
    mps.leftCanonicalise(i);
    mps.rightCanonicalise(i);

    if (i + 1 == n) break;

    MPO mpo = cMultiPhase(env, n - i);
    mps.apply(mpo, i);

    // for (auto j = i + 1; j < n; ++j) {
    //   auto k = j - i + 1;
    //   auto c = std::exp(2i * M_PI / std::pow(2, k));
    //   MPO cp_mpo = controlledGate(env, k, { 1, 0, 0, c });
    //   mps.apply(cp_mpo, i);
    // }
  }
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto N_SITES = 4UL;
  auto CHI_DIS = 2UL;
  auto CHI_LOC = 2UL;

  if (argc > 1) {
    N_SITES = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
  }
  if (argc > 3) {
    CHI_DIS = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
    CHI_LOC = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }

  QTNHEnv env;
  MPS mps(env, N_SITES, SITE_DIM, { CHI_DIS, CHI_LOC });

  utils::barrier();
  auto start = high_resolution_clock::now();

  qft(env, mps);

  utils::barrier();
  auto stop = high_resolution_clock::now();

  MPS zero_amp(env, N_SITES, SITE_DIM, { 1, 1 });
  auto norm = mps.norm();
  auto amp0 = mps.overlap(zero_amp);
  auto delta = duration_cast<milliseconds>(stop - start);
  
  if (utils::is_root()) {
    std::cout << "|T| = " << norm << "\n";
    std::cout << "T[0] = " << amp0 << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }

  // mps.print();
  // std::move(mps).toDense()->print_serial("Res");
}
