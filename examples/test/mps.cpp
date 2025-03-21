#include <algorithm>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  constexpr auto N_SITES = 4UL;
  constexpr auto SITE_DIM = 2UL;
  constexpr auto CHI_DIS = 2UL;
  constexpr auto CHI_LOC = 2UL;

  MPS mps(env, N_SITES, SITE_DIM, { CHI_DIS, CHI_LOC });

  mps.print();

  std::vector<tel> els = {
    1,  1,  1,  1, 
    1, -1,  1, -1, 
    1,  1, -1, -1, 
    1, -1, -1,  1
  };

  std::transform(els.begin(), els.end(), els.begin(), [](auto e) { return e / std::sqrt(2); });

  auto op = SymmTensor::make(env, {}, tidx_tup(4, SITE_DIM), std::move(els));

  utils::barrier();

  if (utils::is_root()) std::cout << "APPLYING OPERATOR\n";
  mps.apply(Tensor::cast<SymmTensor>(op->copy()), { 1, 2 });
  mps.apply(Tensor::cast<SymmTensor>(op->copy()), { 1, 2 });

  mps.print();

  tptr tmps = std::move(mps).toDense();
  tmps->print_serial("TMPS");

  return 0;
}
