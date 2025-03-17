#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  constexpr auto N_SITES = 10UL;
  constexpr auto SITE_DIM = 2UL;
  constexpr auto CHI_DIS = 2UL;
  constexpr auto CHI_LOC = 2UL;

  MPS mps(env, N_SITES, SITE_DIM, { CHI_DIS, CHI_LOC });
  for (auto i = 0UL; i < mps.nSites(); ++i) {
    std::cout << "P" << env.proc_id << " | T" << i << " = " << mps.at(i) << "\n"; 
  }

  return 0;
}
