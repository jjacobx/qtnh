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
