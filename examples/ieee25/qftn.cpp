#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

void qft_triangle(const QTNHEnv& env, TensorNetwork& tn, MPS& mps) {
  auto n = mps.nSites();

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

    tn.insert(std::move(tp));
  }

  for (auto i = 1UL; i + 1 < n; ++i) {
    MPO mpo = qops::cmp(env, n - i);
    mpo.rightCanonicalise();

    tptr tp_h = qops::h(env);

    // Insert operators. 
    for (auto j = 0UL; j < mpo.nSites(); ++j) {
      tptr tp = mpo.at(j).copy();

      // Special cases when first/last operator. 
      if (j == 0UL) {
        tidx_tup loc_shape { tp->locDims().at(0), tp->locDims().at(1), 1, tp->locDims().at(2) };
        tp->reshape({}, loc_shape);
      } else if (j == mpo.nSites() - 1) {
        tidx_tup loc_shape { tp->locDims().at(0), tp->locDims().at(1), tp->locDims().at(2), 1 };
        tp->reshape({}, loc_shape);
      }

      // tp->print_serial("TP");

      // TODO: Check convention. 
      tp = Tensor::permute(std::move(tp), { 3, 1, 0, 2 });

      // Add Hadamard gate for second operator. 
      if (j == 1UL) {
        pcon con(std::move(tp), std::move(tp_h), ConParams({{ 1, 0 }}));
        tp = con.contract();
      }

      tn.insert(std::move(tp));
    }
  }
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
  qft_triangle(env, tn, mps);
  if (utils::is_root()) tn.print();
}