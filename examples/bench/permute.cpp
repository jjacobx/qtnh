#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::qops;
using namespace std::chrono;
using namespace std::complex_literals;

int main(int argc, char* argv[]) {
  using namespace qtnh;

  auto NQUBITS = 5U;
  auto DQUBITS = 2U;
  auto NITER = 10U;

  if (argc > 1) {
    NQUBITS = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
  } 
  if (argc > 2) {
    DQUBITS = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }
  if (argc > 3) {
    NITER = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }

  QTNHEnv env;
  TensorNetwork tn;

  tidx_tup t1_loc_dims(NQUBITS, 2);
  std::vector<tel> t1_els(1 << NQUBITS);
  if (utils::is_root()){
    std::iota(t1_els.begin(), t1_els.end(), 0);
  }

  tptr tp1 = DenseTensor::make(env, {}, t1_loc_dims, std::move(t1_els));
  for (auto i = 0U; i < DQUBITS; ++i) {
    tp1 = Tensor::rescatter(std::move(tp1), 1);
  }

  std::vector<tidx_tup_st> ptup(NQUBITS);
  std::iota(ptup.begin(), ptup.end(), 0);
  // ptup.at(NQUBITS - 1) = 0;

  std::mt19937 gen(2570);
  std::uniform_int_distribution<> dis(0, NQUBITS);
  for (auto i = 0U; i <= 5; ++i) {
    auto ks = std::array<int, 3>{ dis(gen), dis(gen), dis(gen) };
    std::sort(ks.begin(), ks.end());
    std::rotate(ptup.begin() + ks.at(0), ptup.begin() + ks.at(1), ptup.begin() + ks.at(2));

    // if (utils::is_root()) {
    //   std::cout << "ks = " << ks << "\n";
    //   std::cout << "ptup = " << ptup << "\n";
    // }
  }

  if (utils::is_root()) {
    std::cout << "ptup = " << ptup << "\n";
  }

  // std::shuffle(ptup.begin(), ptup.end(), gen);

  utils::barrier();
  auto start = high_resolution_clock::now();

  for (int i = 0U; i < NITER; ++i) {
    tp1 = Tensor::permute(std::move(tp1), ptup);
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  auto delta = duration_cast<milliseconds>(stop - start);
  if (utils::is_root()) {
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }
}