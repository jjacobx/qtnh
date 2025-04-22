#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

using grid_t = std::map<std::pair<std::size_t, std::size_t>, qtnh::uint>;

grid_t qft_triangle(const QTNHEnv& env, TensorNetwork& tn, uint tid_state) {
  auto n = tn.tensor(tid_state)->totDims().size();
  auto d = tn.tensor(tid_state)->disDims().size();
  grid_t grid;

  for (auto i = 0UL; i + 1 < n; ++i) {
    MPO mpo = qops::cmp(env, n - i);

    // Insert operators. 
    auto m = mpo.nSites();
    for (auto j = 0UL; j < m; ++j) {
      tptr tp = mpo.at(j).copy();

      // Special cases when first/last operator. 
      if (j == 0UL) {
        tidx_tup loc_shape { tp->locDims().at(0), tp->locDims().at(1), 1, tp->locDims().at(2) };
        tp->reshape({}, loc_shape);
      } else if (j == m - 1) {
        tidx_tup loc_shape { tp->locDims().at(0), tp->locDims().at(1), tp->locDims().at(2), 1 };
        tp->reshape({}, loc_shape);
      }

      // Add Hadamard gates. 
      if (i == 0UL && j == 0UL) {
        pcon con(std::move(tp), qops::h(env), ConParams({{ 0, 0 }}));
        tp = con.contract();
      } else if (j == 1UL) {
        pcon con(std::move(tp), qops::h(env), ConParams({{ 1, 0 }}));
        tp = con.contract();
      }

      if (i + j < d) {
        tp = Tensor::rescatter(std::move(tp), 2);
      }

      auto tid = tn.insert(std::move(tp));
      grid.insert({ { m - j - 1, i }, tid });
    }
  }

  // Connect tensors with bonds. 
  for (auto& [p, i] : grid) {
    if (p.second == 0) {
      tn.addBond(tid_state, i, {{ n - p.first - 1, 0 }});
    }
    if (grid.find({ p.first + 1, p.second }) != grid.end()) {
      tn.addBond(i, grid.at({ p.first + 1, p.second }), {{ 2, 3 }});
    }
    if (grid.find({ p.first, p.second + 1 }) != grid.end()) {
      tn.addBond(i, grid.at({ p.first, p.second + 1 }), {{ 1, 0 }});
    }
  }

  return grid;
}

uint contract_multi(TensorNetwork& tn, grid_t& grid, uint tid_state, std::size_t multi = 2UL) {
  auto n = tn.tensor(tid_state)->totDims().size();
  grid_t grid2;

  for (auto i = 0UL; i + 1 < n; ++i) {
    for (auto j = 0UL; j < n - i; j += multi) {
      auto tid = grid.at({ j, i });

      for (auto k = 1UL; k < multi && grid.find({ j + k, i }) != grid.end(); ++k) {
        tid = tn.contractTensors(tid, grid.at({ j + k, i }));
      }

      grid2.insert({ { j / multi, i }, tid });
    }
  }

  auto nts = tn.tensorIDs().size() - 1;
  auto k = 0UL;

  for (auto i = 0UL; i + 1 < n; ++i) {
    auto nj = (n - i + multi - 1) / multi;
    for (auto j = 0UL; j < nj; ++j) {
      tid_state = tn.contractTensors(tid_state, grid2.at({ nj - j - 1, i }));

      ++k;
      if (utils::is_root()) std::cout << "Contracted " << k << "/" << nts << " tensors\n";
    }
  }

  return tid_state;
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto DQ = 2UL;
  auto LQ = 2UL;
  auto MULTI = 2UL;

  if (argc > 2) {
    DQ = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
    LQ = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }
  if (argc > 3) {
    MULTI = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }

  QTNHEnv env;

  std::vector<tel> tels(1 << LQ, std::pow(2.0, -0.5 * (DQ + LQ)));
  tptr tp_state = DenseTensor::make(env, tidx_tup(DQ, 2), tidx_tup(LQ, 2), std::move(tels));

  TensorNetwork tn;
  auto tid_state = tn.insert(std::move(tp_state));
  auto grid = qft_triangle(env, tn, tid_state);


  utils::barrier();
  auto start = high_resolution_clock::now();

  tid_state = contract_multi(tn, grid, tid_state, MULTI);

  utils::barrier();
  auto stop = high_resolution_clock::now();
  auto delta = duration_cast<milliseconds>(stop - start);

  tptr tp = tn.extract(tid_state);
  tp->reshape(tidx_tup(DQ, 2), tidx_tup(LQ, 2));
  if (utils::is_root()) {
    std::cout << "\n";
    std::cout << "T[0] = " << (*tp)[0] << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }
}
