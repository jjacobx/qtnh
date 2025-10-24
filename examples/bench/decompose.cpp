#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::qops;
using namespace std::chrono;

int main(int argc, char* argv[]) {
  // Read arguments. 
  auto get_arg = [argv](std::size_t i) {
    return static_cast<unsigned int>(strtol(argv[i], nullptr, 0));
  };

  if (argc != 8) {
    throw std::invalid_argument("Expected 7 numeric arguments.");
  }

  std::array CQS { get_arg(1), get_arg(2) };
  std::array DQS { get_arg(3), get_arg(4) };
  std::array BQS { get_arg(5), get_arg(6) };
  std::array LQS { CQS.at(0) + BQS.at(0), CQS.at(1) + BQS.at(1) };
  auto IT = get_arg(7);

  QTNHEnv env;
  TensorNetwork tn;

  std::mt19937 gen(2025 + env.proc_id);
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  // Compute tensor dimensions. 
  auto dis_dims = utils::concat_dims(tidx_tup(DQS.at(0), 2), tidx_tup(DQS.at(1), 2));
  auto loc_dims = utils::concat_dims(tidx_tup(LQS.at(0), 2), tidx_tup(LQS.at(1), 2));

  // Initialise random tensor elements. 
  std::vector<tel> els;
  if (env.proc_id < utils::dims_to_size(dis_dims)) {
    els = std::vector<tel>(utils::dims_to_size(loc_dims));
    for (auto& el : els) {
      el = tel { dis(gen), dis(gen) };
    }
  }

  tptr tp = DenseTensor::make(env, dis_dims, loc_dims, std::move(els));

  DecParams dp {
  { DQS.at(0), DQS.at(1) }, 
  { LQS.at(0), LQS.at(1) }, 
  { CQS.at(0), CQS.at(1) }, 
  { DQS.at(0), DQS.at(1) }, 
  { BQS.at(0), BQS.at(1) }
};

  utils::barrier();
  auto start = high_resolution_clock::now();

  for (int i = 0U; i < IT; ++i) {
    // TODO: what is the cost of copying tensors? 
    Decomposer dec(tp->copy(), dp, true);
    dec.decompose();

    auto [tp_u, tp_s, tp_v] = dec.extract_results();
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  duration<double, std::milli> delta = stop - start;
  if (utils::is_root()) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time per iteration: " << delta.count() / IT << " ms\n";
  }
}
