#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  constexpr auto N_SITES = 10UL;
  constexpr auto SITE_DIM = 2UL;

  MPS mps(env, N_SITES, SITE_DIM);
  for (auto i = 0UL; i < mps.nSites(); ++i) {
    std::cout << "T" << i << " = " << mps.at(i) << "\n"; 
  }

  return 0;
}
