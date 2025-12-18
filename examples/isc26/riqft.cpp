#include <chrono>
#include <iostream>
#include <ostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;

constexpr auto SITE_DIM = 2UL;

void qft(const QTNHEnv& env, BCMPS& mps, bool update_dims = false) {
  auto n = mps.nSites();

  for (auto i = 0UL; i < n; ++i) {
    if (utils::is_root()) {
      std::cout << "Iteration " << i + 1 << "/" << n << std::flush;
    }

    auto start = high_resolution_clock::now();

    mps.apply(qops::h(env), { i }, update_dims);
    mps.leftCanonicalise(i);
    mps.rightCanonicalise(i);

    if (i + 1 == n) break;

    MPO mpo = qops::cmp(env, n - i);
    mpo.rightCanonicalise();
    mps.apply(mpo, i, update_dims);

    auto stop = high_resolution_clock::now();
    auto delta = duration_cast<milliseconds>(stop - start);

    if (utils::is_root()) {
      std::cout << " (" << delta.count() << " ms)" << std::endl;
    }
  }
}

void iqft(const QTNHEnv& env, BCMPS& mps, bool update_dims = false) {
  auto n = mps.nSites();

  mps.leftCanonicalise(n - 1);
  mps.apply(qops::h(env), { n - 1 }, update_dims);
  for (auto i = 1UL; i < n; ++i) {
    if (utils::is_root()) {
      std::cout << "Applying MPO " << i << "/" << n - 1 << std::flush;
    }

    auto start = high_resolution_clock::now();

    MPO mpo = qops::cmp(env, i + 1, -1.0);
    mpo.rightCanonicalise();

    mps.leftCanonicalise(n - i - 1);
    mps.rightCanonicalise(n - i - 1);

    auto checkpoint = high_resolution_clock::now();
    auto delta1 = duration_cast<milliseconds>(checkpoint - start);
    if (utils::is_root()) {
      std::cout << " (" << delta1.count() << " ms + " << std::flush;
    }

    mps.apply(mpo, n - i - 1, update_dims);
    mps.apply(qops::h(env), { n - i - 1 }, update_dims);

    auto stop = high_resolution_clock::now();
    auto delta2 = duration_cast<milliseconds>(stop - start);
    if (utils::is_root()) {
      std::cout << delta2.count() << " ms)" << std::endl;
    }
  }
}

int main(int argc, char* argv[]) {
  QTNHEnv env;
  
  auto N_SITES = 10UL;
  auto CHI_CYC = 4UL;
  auto CHI_DIS = 2UL;
  auto CHI_BLK = 2UL;
  auto CHI_SAT = 0.1;

  enum class DecMethod { SVD, QPD };
  auto DEC_METHOD = DecMethod::SVD;

  if (argc > 1) {
    N_SITES = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
  }
  if (argc > 4) {
    CHI_CYC = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
    CHI_DIS = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
    CHI_BLK = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
  }
  if (argc > 5) {
    CHI_SAT = strtod(argv[5], nullptr);
  }
  if (argc > 6) {
    std::string method(argv[6]);
    if (method == "SVD") {
      DEC_METHOD = DecMethod::SVD;
      if (utils::is_root()) std::cout << "Using SVD decomposition method.\n";
    } else if (method == "QPD") {
      DEC_METHOD = DecMethod::QPD;
      if (utils::is_root()) std::cout << "Using QPD decomposition method.\n";
    } else if (utils::is_root()) {
      std::cout << "Unknown decomposition method, defaulting to SVD.\n";
    }
  }

  auto chi_in = static_cast<std::size_t>(CHI_CYC * CHI_DIS * CHI_BLK * CHI_SAT);

  if (utils::is_root()) {
    std::cout << "QFT with n = " << N_SITES << std::endl;
    std::cout << "CHI = " << CHI_CYC * CHI_DIS * CHI_BLK << std::endl;
    std::cout << "CHI_IN = " << chi_in << " (" << CHI_SAT * 100 << "%)" << std::endl;
  }
  
  auto mps = BCMPS::rand(env, N_SITES, SITE_DIM, { CHI_CYC, CHI_DIS, CHI_BLK }, chi_in);

  // * Validation. 
  // auto mps1 = mps.copy();
  // auto mps2 = mps.copy();
  // iqft(env, mps1);
  // qft(env, mps1);
  // auto overlap12 = mps1.overlap(mps2);
  // if (utils::is_root()) {
  //   std::cout << "Validation overlap = " << overlap12 << std::endl;
  // }

  utils::barrier();
  auto start = high_resolution_clock::now();

  switch (DEC_METHOD) {
    case DecMethod::SVD:
      iqft(env, mps, true);
      break;
    case DecMethod::QPD:
      iqft(env, mps, false);
      break;
    default:
      utils::throw_unimplemented();
      break;
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  BCMPS zero_amp(env, N_SITES, SITE_DIM, { 1, 1, 1 });
  auto norm = mps.norm();
  auto bonds = mps.bondDims();

  mps.renormalise();
  auto amp0 = mps.overlap(zero_amp);
  auto delta = duration_cast<milliseconds>(stop - start);
  
  if (utils::is_root()) {
    std::cout << std::endl;
    std::cout << "|T| = " << norm.real() << std::endl;
    std::cout << "T[0] = " << amp0 << std::endl;
    std::cout << "Max chis = " << bonds << std::endl;
    std::cout << "Time taken: " << delta.count() << " ms" << std::endl;
  }
}
