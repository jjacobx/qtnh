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

  // std::vector<qtnh::wire> sorted_targets;
  // auto a = targets.size() % (n / 2);
  // for (auto i = 0UL; i < targets.size(); ++i) {
  //   auto b = i * (n / 2);
  //   auto c = b / targets.size();
  //   auto j = (b - a * c) % targets.size() + c;
    
  //   sorted_targets.push_back(targets.at(j));
  // }
  
  // return sorted_targets;

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

void rcs(const QTNHEnv& env, BCMPS& mps, std::size_t m, std::size_t n, std::size_t d, 
         const std::vector<FSimPattern>& patterns, bool update_bonds = true) {
  auto mn = mps.nSites();

  std::mt19937 gen(2025);
  std::uniform_int_distribution<std::size_t> dist02(0, 2);

  for (auto i = 0UL; i < d; ++i) {
    if (utils::is_root()) {
      std::cout << "\nIteration " << i + 1 << "/" << d << std::endl;
      std::cout << "Applying random gates..." << std::endl;
    }

    auto start = high_resolution_clock::now();
    
    for (auto j = 0UL; j < mn; ++j) {
      auto num = dist02(gen);
      mps.apply(rand_gate(num)(env), { j });
    }

    auto stop = high_resolution_clock::now();
    auto delta = duration_cast<milliseconds>(stop - start);

    auto targets = fsim_targets(patterns.at(i % patterns.size()), m, n);

    if (utils::is_root()) {
      std::cout << "Done (" << delta.count() << " ms)" << std::endl;
      std::cout << "Applying entangling gates..." << std::endl;
      std::cout << "Targets: " << targets << std::endl;
    }

    auto counter = 1UL;
    for (auto ts : targets) {
      auto start = high_resolution_clock::now();

      MPO mpo = qops::fsim(env, ts.second - ts.first + 1);
      mpo.rightCanonicalise();
      
      // * Full recanonicalisation might slightly improve accuracy. 
      // mps.leftCanonicalise(mps.nSites() - 1);
      mps.leftCanonicalise(ts.second);
      mps.rightCanonicalise(ts.first);

      auto checkpoint = high_resolution_clock::now();

      mps.apply(mpo, ts.first, update_bonds);

      auto stop = high_resolution_clock::now();
      auto delta12 = duration_cast<milliseconds>(checkpoint - start);
      auto delta23 = duration_cast<milliseconds>(stop - checkpoint);
      auto delta13 = duration_cast<milliseconds>(stop - start);

      if (utils::is_root()) {
        std::cout << counter++ << "/" << targets.size() << " done (" << 
          delta13.count() << " ms = " << delta12.count() << " + " << 
          delta23.count() << " ms)" << std::endl;
      }
    }

    auto norm = mps.norm();
    if (utils::is_root()) {
      std::cout << "bonds = " << mps.bondDims() << "\n";
      std::cout << "norm = " << norm << "\n";
    }
  }
}

int main(int argc, char* argv[]) {
  QTNHEnv env;

  constexpr auto SITE_DIM = 2UL;
  
  auto NROW = 4UL;
  auto NCOL = 4UL;
  auto DEPTH = 4UL;
  auto CHI_DIS = 2UL;
  auto CHI_LOC = 8UL;

  auto CHI_CYC = 4UL;
  auto CHI_BLK = 2UL;

  enum class DecMethod { SVD, QPD };
  auto DEC_METHOD = DecMethod::SVD;

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
  if (argc > 6) {
    CHI_CYC = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
    CHI_DIS = static_cast<unsigned int>(strtol(argv[5], nullptr, 0));
    CHI_BLK = static_cast<unsigned int>(strtol(argv[6], nullptr, 0));
    
    CHI_LOC = CHI_CYC * CHI_BLK;
  }
  if (argc > 7) {
    std::string method(argv[7]);
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

  if (utils::is_root()) {
    std::cout << "RCS (" << NROW << ", " << NCOL << ") with d = " << DEPTH << "\n";
    std::cout << "CHI = " << CHI_DIS * CHI_LOC << "\n";
  }
  
  // MPS mps(env, NROW * NCOL, SITE_DIM, { CHI_DIS, CHI_LOC });
  BCMPS mps(env, NROW * NCOL, SITE_DIM, { CHI_CYC, CHI_DIS, CHI_BLK });
  // auto mps = BCMPS::rand(env, NROW * NCOL, SITE_DIM, { CHI_CYC, CHI_DIS, CHI_BLK }, CHI_CYC * CHI_DIS * CHI_BLK);

  std::vector<FSimPattern> patterns {
    FSimPattern::A, FSimPattern::B, FSimPattern::C, FSimPattern::D, 
    FSimPattern::C, FSimPattern::D, FSimPattern::A, FSimPattern::B
  };
  // std::vector<FSimPattern> patterns {
  //   FSimPattern::C, FSimPattern::D
  // };

  utils::barrier();
  auto start = high_resolution_clock::now();

  switch (DEC_METHOD) {
    case DecMethod::SVD:
      rcs(env, mps, NROW, NCOL, DEPTH, patterns, true);
      break;
    case DecMethod::QPD:
      rcs(env, mps, NROW, NCOL, DEPTH, patterns, false);
      break;
    default:
      utils::throw_unimplemented();
      break;
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();
  // MPS zero_amp(env, NROW * NCOL, SITE_DIM, { 1, 1 });
  BCMPS zero_amp(env, NROW * NCOL, SITE_DIM, { 1, 1, 1 });
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