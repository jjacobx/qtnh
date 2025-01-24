#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;

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

  tidx_tup t1_loc_dims(10, NQUBITS);
  std::vector<tel> t1_els(1 << NQUBITS);
  std::iota(t1_els.begin(), t1_els.end(), 0);

  tptr tp1 = DenseTensor::make(env, {}, t1_loc_dims, std::move(t1_els));
  tp1 = Tensor::rescatter(std::move(tp1), int(DQUBITS));

  std::vector<tidx_tup_st> ptup(NQUBITS);
  std::iota(ptup.begin(), ptup.end(), 1);
  ptup.at(NQUBITS - 1) = 0;

  utils::barrier();
  auto start = high_resolution_clock::now();

  for (int i = 0U; i < NITER; ++i) {
    tp1 = Tensor::permute(std::move(tp1), ptup);
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  auto delta = duration_cast<microseconds>(stop - start);
  if (utils::is_root()) {
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }
}