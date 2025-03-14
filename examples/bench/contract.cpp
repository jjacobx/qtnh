#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
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

  tptr Psi = DenseTensor::make(env, {}, t1_loc_dims, std::move(t1_els));
  for (auto i = 0U; i < DQUBITS; ++i) {
    Psi = Tensor::rescatter(std::move(Psi), 1);
  }

  tptr X = DenseTensor::make(env, {}, { 2, 2 }, { 0.0, 1.0, 1.0, 0.0 });

  utils::barrier();
  auto start = high_resolution_clock::now();

  for (int i = 0U; i < NITER; ++i) {
    auto params = ConParams(std::vector<wire> {{ NQUBITS - 1, 0 }});
    auto con = pcon(std::move(Psi), X->copy(), params);
    Psi = con.contract();
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  auto delta = duration_cast<milliseconds>(stop - start);
  if (utils::is_root()) {
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }
}