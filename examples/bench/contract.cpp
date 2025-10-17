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

  std::array DQS { get_arg(1), get_arg(2), get_arg(3) };
  std::array LQS { get_arg(4), get_arg(5), get_arg(6) };
  auto IT = get_arg(7);

  QTNHEnv env;
  TensorNetwork tn;

  std::mt19937 gen(2025 + env.proc_id);
  std::uniform_real_distribution<> dis(-1.0, 1.0);

  // Compute tensor dimensions. 
  auto dis_dims1 = utils::concat_dims(tidx_tup(DQS.at(0), 2), tidx_tup(DQS.at(1), 2));
  auto dis_dims2 = utils::concat_dims(tidx_tup(DQS.at(1), 2), tidx_tup(DQS.at(2), 2));
  auto loc_dims1 = utils::concat_dims(tidx_tup(LQS.at(0), 2), tidx_tup(LQS.at(1), 2));
  auto loc_dims2 = utils::concat_dims(tidx_tup(LQS.at(1), 2), tidx_tup(LQS.at(2), 2));

  // Initialise random tensor elements. 
  std::vector<tel> els1;
  if (env.proc_id < utils::dims_to_size(dis_dims1)) {
    els1 = std::vector<tel>(utils::dims_to_size(loc_dims1));
    for (auto& el : els1) {
      el = tel { dis(gen), dis(gen) };
    }
  }

  std::vector<tel> els2;
  if (env.proc_id < utils::dims_to_size(dis_dims2)) {
    els2 = std::vector<tel>(utils::dims_to_size(loc_dims2));
    for (auto& el : els2) {
      el = tel { dis(gen), dis(gen) };
    }
  }
  
  tptr tp1 = DenseTensor::make(env, dis_dims1, loc_dims1, std::move(els1));
  tptr tp2 = DenseTensor::make(env, dis_dims2, loc_dims2, std::move(els2));

  // Compute contraction wires. 
  std::vector<wire> ws;
  for (auto i = 0U; i < DQS.at(1); ++i) {
    ws.push_back({ DQS.at(0) + i , i });
  }
  for (auto i = 0U; i < LQS.at(1); ++i) {
    ws.push_back({ DQS.at(0) + DQS.at(1) + LQS.at(0) + i, DQS.at(1) + DQS.at(2) + i });
  }

  utils::barrier();
  auto start = high_resolution_clock::now();

  // Contraction pattern:
  // (d0, d1; l0, l1) x (d1, d2; l1, l2) = (d0, d2; l0, l2)
  for (int i = 0U; i < IT; ++i) {
    // TODO: what is the cost of copying tensors? 
    auto con = pcon(tp1->copy(), tp2->copy(), ConParams(ws));
    tptr tp3 = con.contract();
  }

  utils::barrier();
  auto stop = high_resolution_clock::now();

  duration<double, std::milli> delta = stop - start;
  if (utils::is_root()) {
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Time per iteration: " << delta.count() / IT << " ms\n";
  }
}
