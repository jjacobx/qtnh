#include <chrono>
#include <iostream>
#include <map>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

auto qft_triangle(const QTNHEnv& env, TensorNetwork& tn, MPS& mps) {
  auto n = mps.nSites();
  std::map<std::pair<std::size_t, std::size_t>, qtnh::uint> grid;

  // Prepare initial state. 
  mps.apply(qops::h(env), { 0 });
  mps.rightCanonicalise(0);

  MPO mpo = qops::cmp(env, n);
  mpo.rightCanonicalise();
  mps.apply(mpo, 0);

  // Insert initial state. 
  for (auto i = 0UL; i < n; ++i) {
    tptr tp = mps.site(i).copy();
    auto k = mps.totChi();

    tp = Tensor::rescatter(std::move(tp), -2);
    tp = Tensor::permute(std::move(tp), { 0, 3, 2, 1, 4 });
    tp->reshape({}, { mps.totChi(), mps.siteDims().at(i), mps.totChi(), 1 });

    // Special cases when first/last site. 
    if (i == 0UL) { 
      tp = Tensor::truncate(std::move(tp), 0, 1);
    } else if (i == n - 1) {
      tp = Tensor::truncate(std::move(tp), 2, 1);
    }

    // Another special case – Hadamard gate. 
    if (i == 1UL) {
      pcon con(std::move(tp), qops::h(env), ConParams({{ 1, 0 }}));
      tp = con.contract();
    }

    auto tid = tn.insert(std::move(tp));
    grid.insert({ { n - i - 1, 0 }, tid });
  }

  for (auto i = 1UL; i + 1 < n; ++i) {
    MPO mpo = qops::cmp(env, n - i);
    mpo.rightCanonicalise();

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

      // TODO: Check convention. 
      tp = Tensor::permute(std::move(tp), { 3, 1, 0, 2 });

      // Add Hadamard gate for second operator. 
      if (j == 1UL) {
        pcon con(std::move(tp), qops::h(env), ConParams({{ 1, 0 }}));
        tp = con.contract();
      }

      auto tid = tn.insert(std::move(tp));
      grid.insert({ { m - j - 1, i }, tid });
    }
  }

  // Connect tensors with bonds. 
  for (auto& [p, i] : grid) {
    if (grid.find({ p.first + 1, p.second }) != grid.end()) {
      tn.addBond(i, grid.at({ p.first + 1, p.second }), {{ 0, 2 }});
    }
    if (grid.find({ p.first, p.second + 1 }) != grid.end()) {
      tn.addBond(i, grid.at({ p.first, p.second + 1 }), {{ 1, 3 }});
    }
  }

  return grid;
}

int main(int argc, char* argv[]) {
  constexpr auto SITE_DIM = 2UL;
  
  auto N_SITES = 4UL;
  auto CHI_DIS = 2UL;
  auto CHI_LOC = 2UL;

  if (argc > 1) {
    N_SITES = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
  }
  if (argc > 3) {
    CHI_DIS = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
    CHI_LOC = static_cast<unsigned int>(strtol(argv[3], nullptr, 0));
  }

  QTNHEnv env;
  MPS mps(env, N_SITES, SITE_DIM, { CHI_DIS, CHI_LOC });

  TensorNetwork tn;
  auto grid = qft_triangle(env, tn, mps);
  if (utils::is_root()) tn.print();

  if (N_SITES == 4UL) {
    auto tid11 = tn.contractTensors(4, 7);
    auto tid12 = tn.contractTensors(3, 6);
    auto tid1 = tn.contractTensors(tid11, tid12);

    auto tid21 = tn.contractTensors(2, 5);
    auto tid2 = tn.contractTensors(tid21, 1);

    auto tid3 = tn.contractTensors(8, 9);

    auto tid4 = tn.contractTensors(tid1, tid2);
    auto tid_final = tn.contractTensors(tid3, tid4);

    tptr tp = tn.extract(tid_final);
    tp->print_serial("Final");
  }
}
