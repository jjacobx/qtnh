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
    for (auto j = 0UL; j < mn; ++j) {
      mps.apply(rand_gate(dist02(gen))(env), { j });
    }

    auto targets = fsim_targets(patterns.at(i % patterns.size()), m, n);
    if (utils::is_root()) std::cout << targets << "\n";

    for (auto ts : targets) {
      MPO mpo = qops::fsim(env, ts.second - ts.first + 1);
      mpo.rightCanonicalise();
      // mpo.print();
      
      mps.leftCanonicalise(mps.nSites() - 1);
      mps.rightCanonicalise(ts.first);
      mps.apply(mpo, ts.first);
      
      auto norm = mps.norm();
      if (utils::is_root()) std::cout << "bonds = " << mps.bondDims() << "\n";
      if (utils::is_root()) std::cout << "norm = " << norm << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto NROW = 4UL;
  auto NCOL = 4UL;
  auto DEPTH = 4UL;
  auto CHI_DIS = 2UL;
  auto CHI_LOC = 2UL;

  if (argc > 2) {
    NROW = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
    NCOL = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }
  if (argc > 3) {
    DEPTH = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }
  if (argc > 5) {
    CHI_DIS = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
    CHI_LOC = static_cast<unsigned int>(strtol(argv[5], nullptr, 0));
  }

  QTNHEnv env;
  MPS mps(env, NROW * NCOL, SITE_DIM, { CHI_DIS, CHI_LOC });
  std::vector<FSimPattern> patterns {
    FSimPattern::A, FSimPattern::B, FSimPattern::C, FSimPattern::D, 
    FSimPattern::C, FSimPattern::D, FSimPattern::A, FSimPattern::B
  };

  utils::barrier();
  auto start = high_resolution_clock::now();

  rcs(env, mps, NROW, NCOL, DEPTH, patterns);

  utils::barrier();
  auto stop = high_resolution_clock::now();

  mps.leftCanonicalise(mps.nSites() - 1);
  mps.rightCanonicalise(0);

  MPS zero_amp(env, NROW * NCOL, SITE_DIM, { 1, 1 });
  auto norm = mps.norm();
  auto bonds = mps.bondDims();
  auto amp0 = mps.overlap(zero_amp);
  auto delta = duration_cast<milliseconds>(stop - start);
  
  if (utils::is_root()) {
    std::cout << "|T| = " << norm << "\n";
    std::cout << "T[0] = " << amp0 << "\n";
    std::cout << "Max chis = " << bonds << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }

  // mps.print();
  // std::move(mps).toDense()->print_serial("Res");
}