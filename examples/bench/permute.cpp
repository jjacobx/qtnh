#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::qops;
using namespace std::chrono;
using namespace std::complex_literals;

int main(int argc, char* argv[]) {
  using namespace qtnh;

  auto NQ = 5U;
  auto DQ = 2U;
  auto IT = 10U;

  if (argc > 1) {
    NQ = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
  } 
  if (argc > 2) {
    DQ = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }
  if (argc > 3) {
    IT = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }

  QTNHEnv env;
  TensorNetwork tn;
  std::mt19937 gen(2025);

  // Create distributed tensor filled in with contiguous numbers. 
  tidx_tup dis_dims(DQ, 2);
  tidx_tup loc_dims(NQ - DQ, 2);
  auto loc_size = utils::dims_to_size(loc_dims);
  auto dis_size = utils::dims_to_size(dis_dims);

  std::vector<tel> els(loc_size);
  if (env.proc_id < dis_size){
    std::iota(els.begin(), els.end(), env.proc_id * loc_size);
  }

  tptr tp = DenseTensor::make(env, dis_dims, loc_dims, std::move(els));

  utils::barrier();
  auto start = high_resolution_clock::now();

  for (int i = 0U; i < IT; ++i) {
    // Create semi-random permutation tuple: 
    // - select random range
    // - rotate it by random amount
    // - repeat 5 times
    std::vector<tidx_tup_st> ptup(NQ);
    std::iota(ptup.begin(), ptup.end(), 0);
    
    std::uniform_int_distribution<> dis(0, NQ);

    for (auto i = 0U; i < 4; ++i) {
      auto ks = std::array<int, 3>{ dis(gen), dis(gen), dis(gen) };
      std::sort(ks.begin(), ks.end());
      std::rotate(ptup.begin() + ks.at(0), ptup.begin() + ks.at(1), ptup.begin() + ks.at(2));
    }

    tp = Tensor::permute(std::move(tp), ptup);
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  duration<double, std::milli> delta = stop - start;
  if (utils::is_root()) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time per iteration: " << delta.count() / IT << " ms\n";
  }
}
