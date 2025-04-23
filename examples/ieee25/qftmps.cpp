#include <algorithm>
#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

void qft(const QTNHEnv& env, MPS& mps, bool swap_out = false) {
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

  if (swap_out) {
    for (auto i = 0UL; i < n / 2; ++i) {
      MPO mpo = qops::swap(env, n - 2 * i);
      mpo.rightCanonicalise();
  
      mps.leftCanonicalise(i);
      mps.rightCanonicalise(i);
      mps.apply(mpo, i);
    }
  }
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto N_SITES = 4UL;
  auto CHI_DIS = 2UL;
  auto CHI_LOC_FR = 32UL;
  auto CHI_LOC_TO = 16UL;
  auto BOND_DIM = 2UL;

  if (argc > 1) {
    N_SITES = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
  }
  if (argc > 4) {
    CHI_DIS = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
    CHI_LOC_FR = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
    CHI_LOC_TO = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
  }
  if (argc > 5) {
    BOND_DIM = static_cast<unsigned int>(strtol(argv[5], nullptr, 0));
  }

  std::vector<std::size_t> chi_locs;
  for (auto i = CHI_LOC_FR; i >= CHI_LOC_TO; i /= 2) {
    chi_locs.push_back(i);
  }

  QTNHEnv env;

  // Use |0> if BOND_DIM == 0. 
  auto mps_ref = (BOND_DIM > 0)
    ? MPS::rand(env, N_SITES, SITE_DIM, { CHI_DIS, chi_locs.at(0) }, BOND_DIM)
    : MPS(env, N_SITES, SITE_DIM, { CHI_DIS, chi_locs.at(0) });

  utils::barrier();
  auto start = high_resolution_clock::now();

  qft(env, mps_ref);

  utils::barrier();
  auto stop = high_resolution_clock::now();

  // Update all sites. 
  mps_ref.rightCanonicalise(0);

  MPS zero_amp(env, N_SITES, SITE_DIM, { 1, 1 });
  auto norm = mps_ref.norm();
  auto bonds = mps_ref.bondDims();
  auto max = *std::max_element(bonds.begin(), bonds.end());
  auto amp0 = mps_ref.overlap(zero_amp);
  auto delta = duration_cast<milliseconds>(stop - start);
  
  if (utils::is_root()) {
    std::cout << "Results " << chi_locs.at(0) << ":\n";
    std::cout << "|T| = " << norm << "\n";
    std::cout << "T[0] = " << amp0 << "\n";
    std::cout << "Max chis = " << bonds << "\n";
    std::cout << "Max chi = " << max << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
    std::cout << "\n";
  }

  for (auto i = 1UL; i < chi_locs.size(); ++i) {
    auto mps = MPS::rand(env, N_SITES, SITE_DIM, { CHI_DIS, chi_locs.at(i) }, BOND_DIM);

    utils::barrier();
    auto start = high_resolution_clock::now();
  
    qft(env, mps);
  
    utils::barrier();
    auto stop = high_resolution_clock::now();

    // Update all sites. 
    mps.rightCanonicalise(0);

    // Get norm before renormalising, as it resets to 1. 
    auto norm = mps.norm();
    mps.renormalise();

    auto bonds = mps.bondDims();
    auto max = *std::max_element(bonds.begin(), bonds.end());
    auto amp0 = mps.overlap(zero_amp);
    auto overlap = mps_ref.overlap(mps);
    auto delta = duration_cast<milliseconds>(stop - start);

    if (utils::is_root()) {
      std::cout << "Results " << chi_locs.at(i) << ":\n";
      std::cout << "|T| = " << norm << "\n";
      std::cout << "T[0] = " << amp0 << "\n";
      std::cout << "<REF|T> = " << overlap.real() << "\n";
      std::cout << "Max chis = " << bonds << "\n";
      std::cout << "Max chi = " << max << "\n";
      std::cout << "Time taken: " << delta.count() << " ms\n";
      std::cout << "\n";
    }
  }
}
