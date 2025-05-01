#include <chrono>
#include <iostream>
#include <random>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

enum class FSimPattern { A, B, C, D };

std::vector<qtnh::wire> fsim_targets(FSimPattern pattern, std::size_t m, std::size_t n) {
  std::vector<qtnh::wire> targets;

  switch (pattern) {
    case FSimPattern::A:
      for (auto i = 0UL; i < m; ++i) {
        for (auto j = i % 2; j + 1 < n; j += 2) {
          targets.push_back({ n * i + j, n * i + j + 1 });
        }
      } break;
    case FSimPattern::B:
      for (auto i = 0UL; i < m; ++i) {
        for (auto j = 1UL - i % 2; j + 1 < n; j += 2) {
          targets.push_back({ n * i + j, n * i + j + 1 });
        }
      } break;
    case FSimPattern::C:
      for (auto i = 0UL; i + 1 < m; ++i) {
        for (auto j = i % 2; j < n; j += 2) {
          targets.push_back({ n * i + j, n * i + j + n });
        }
      } break;
    case FSimPattern::D:
      for (auto i = 0UL; i + 1 < m; ++i) {
        for (auto j = 1UL - i % 2; j < n; j += 2) {
          targets.push_back({ n * i + j, n * i + j + n });
        }
      } break;
  }
  
  return targets;
}

auto rand_gate(std::size_t i) {
  switch(i) {
    case 0:
      return qops::sqrt_x;
    case 1:
      return qops::sqrt_y;
    case 2:
      return qops::sqrt_w;
    default:
      throw std::invalid_argument("Only three random gates available.");
  }
}

void rcs(const QTNHEnv& env, MPS& mps, std::size_t m, std::size_t n, std::size_t d, 
         const std::vector<FSimPattern>& patterns) {
  auto mn = mps.nSites();

  std::mt19937 gen(2025);
  std::uniform_int_distribution<std::size_t> dist02(0, 2);

  for (auto i = 0UL; i < d; ++i) {
    if (utils::is_root()) {
      std::cout << "Iteration " << i + 1 << "/" << d << "\n";
    }

    for (auto j = 0UL; j < mn; ++j) {
      mps.apply(rand_gate(dist02(gen))(env), { j });
    }

    auto targets = fsim_targets(patterns.at(i % patterns.size()), m, n);
    if (utils::is_root()) {
      std::cout << targets << "\n";
    }

    for (auto ts : targets) {
      MPO mpo = qops::fsim(env, ts.second - ts.first + 1);
      mpo.rightCanonicalise();
      
      mps.leftCanonicalise(mps.nSites() - 1);
      mps.rightCanonicalise(ts.first);
      mps.apply(mpo, ts.first);
    }

    auto norm = mps.norm();
    if (utils::is_root()) {
      std::cout << "bonds = " << mps.bondDims() << "\n";
      std::cout << "norm = " << norm << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto NROW = 4UL;
  auto NCOL = 4UL;
  auto DEPTH = 4UL;
  auto DCHI = 2UL;
  auto LCHI_UP = 8UL;
  auto LCHI_DN = 8UL;

  if (argc > 2) {
    NROW = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
    NCOL = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }
  if (argc > 3) {
    DEPTH = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }
  if (argc > 5) {
    DCHI = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
    LCHI_UP = static_cast<unsigned int>(strtol(argv[5], nullptr, 0));
    LCHI_DN = LCHI_UP;
  } if (argc > 6) {
    LCHI_DN = static_cast<unsigned int>(strtol(argv[6], nullptr, 0));
  }

  std::vector<std::size_t> chi_locs;
  for (auto i = LCHI_UP; i >= LCHI_DN; i /= 2) {
    chi_locs.push_back(i);
  }

  QTNHEnv env;
  MPS mps_ref(env, NROW * NCOL, SITE_DIM, { DCHI, LCHI_UP });
  std::vector<FSimPattern> patterns {
    FSimPattern::A, FSimPattern::B, FSimPattern::C, FSimPattern::D, 
    FSimPattern::C, FSimPattern::D, FSimPattern::A, FSimPattern::B
  };

  utils::barrier();
  auto start = high_resolution_clock::now();

  rcs(env, mps_ref, NROW, NCOL, DEPTH, patterns);

  utils::barrier();
  auto stop = high_resolution_clock::now();

  // MPS zero_amp(env, NROW * NCOL, SITE_DIM, { 1, 1 });
  // auto norm = mps.norm();
  // auto bonds = mps.bondDims();
  // auto amp0 = mps.overlap(zero_amp);
  // auto delta = duration_cast<milliseconds>(stop - start);
  
  // if (utils::is_root()) {
  //   std::cout << "|T| = " << norm << "\n";
  //   std::cout << "T[0] = " << amp0 << "\n";
  //   std::cout << "Max chis = " << bonds << "\n";
  //   std::cout << "Time taken: " << delta.count() << " ms\n";
  // }

  // Update all sites. 
  mps_ref.rightCanonicalise(0);

  MPS zero_amp(env, NROW * NCOL, SITE_DIM, { 1, 1 });
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
    std::cout << "\n" << std::flush;
  }

  for (auto i = 1UL; i < chi_locs.size(); ++i) {
    MPS mps(env, NROW * NCOL, SITE_DIM, { DCHI, chi_locs.at(i) });

    utils::barrier();
    auto start = high_resolution_clock::now();
  
    rcs(env, mps, NROW, NCOL, DEPTH, patterns);
  
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
      std::cout << "\n" << std::flush;
    }
  }
}