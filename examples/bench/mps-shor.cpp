#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

// m-bit phase estimation of modular exponentiation of 2^q mod 2^n - 1. 
void order_finding(MPS& mps, std::size_t m, std::size_t n, std::size_t q) {
  auto& env = mps.site(0).bc().env();

  mps.apply(qops::x(env), { m + n - 1 });
  for (auto i = 0UL; i < m; ++i) {
    mps.apply(qops::h(env), { i });

    auto swap_tars = qops::rotate_swaps(n, q * (1 << i));
    for (auto [a, b] : swap_tars) {
      if (utils::is_root()) {
        std::cout << a << ", " << b << "\n";
        std::cout << "Bond dims = " << mps.bondDims() << "\n";
      }
      auto swap_mpo = qops::swap(env, b - a + 1);
      auto cswap_mpo = qops::cmpo(env, swap_mpo, m - i + b + 1);

      mps.apply(cswap_mpo, i);
      mps.leftCanonicalise(m + n - 1);
      mps.rightCanonicalise(i);
    }
  }

  for (auto i = 0UL; i < m; ++i) {
    mps.rightCanonicalise(0);

    if (i > 0) {
      MPO mpo = qops::icmp(env, i + 1);
      mps.apply(mpo, 0);
    }

    mps.apply(qops::h(env), { i });
  }
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto M = 2UL;
  auto N = 3UL;
  auto Q = 1UL;
  auto CHI_DIS = 2UL;
  auto CHI_LOC = 8UL;

  if (argc > 2) {
    M = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
    N = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }
  if (argc > 3) {
    Q = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }
  if (argc > 4) {
    CHI_DIS = static_cast<unsigned int>(strtol(argv[4], nullptr, 0));
    CHI_LOC = static_cast<unsigned int>(strtol(argv[5], nullptr, 0));
  }

  QTNHEnv env;
  MPS mps(env, M + N, SITE_DIM, { CHI_DIS, CHI_LOC });

  utils::barrier();
  auto start = high_resolution_clock::now();

  order_finding(mps, M, N, Q);

  utils::barrier();
  auto stop = high_resolution_clock::now();

  MPS zero_amp(env, M + N, SITE_DIM, { 1, 1 });
  auto norm = mps.norm();
  auto amp0 = mps.overlap(zero_amp);
  auto delta = duration_cast<milliseconds>(stop - start);
  
  if (utils::is_root()) {
    std::cout << "|T| = " << norm << "\n";
    std::cout << "T[0] = " << amp0 << "\n";
    std::cout << "Bond dims = " << mps.bondDims() << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }
}