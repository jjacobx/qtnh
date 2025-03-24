#include <algorithm>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  constexpr auto N_SITES  = 5UL;
  constexpr auto SITE_DIM = 2UL;
  constexpr auto CHI_DIS  = 2UL;
  constexpr auto CHI_LOC  = 3UL;

  MPS mps(env, N_SITES, SITE_DIM, { CHI_DIS, CHI_LOC });

  mps.print();

  std::vector<tel> els_h = {
    1 / std::sqrt(2),  1 / std::sqrt(2), 
    1 / std::sqrt(2), -1 / std::sqrt(2)
  };

  std::vector<tel> els_hh = {
    .5,  .5,  .5,  .5, 
    .5, -.5,  .5, -.5, 
    .5,  .5, -.5, -.5, 
    .5, -.5, -.5,  .5
  };

  std::vector<tel> els_cx = {
    1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 0, 1, 
    0, 0, 1, 0
  };

  auto h =  SymmTensor::make(env, {}, tidx_tup(2, SITE_DIM), std::move(els_h));
  auto hh = SymmTensor::make(env, {}, tidx_tup(4, SITE_DIM), std::move(els_hh));
  auto cx = SymmTensor::make(env, {}, tidx_tup(4, SITE_DIM), std::move(els_cx));

  utils::barrier();

  mps.apply(Tensor::cast<SymmTensor>(h->copy()), { 0 });
  mps.apply(Tensor::cast<SymmTensor>(cx->copy()), { 0, 2 });

  mps.print();

  tptr tmps = std::move(mps).toDense();
  tmps->print_serial("TMPS");

  return 0;
}
