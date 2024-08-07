#ifndef QFT_TENSORS_HPP 
#define QFT_TENSORS_HPP

#include "validation-primitives.hpp"
#include "tensor/network.hpp"
#include "tensor/special.hpp"

using namespace std::complex_literals;

namespace gen {
  tensor_info plus_state {{ 2 }, { std::pow(2, -.5), std::pow(2, -.5) }};
  tensor_info hadamard {{ 2, 2 }, { std::pow(2, -.5),  std::pow(2, -.5), std::pow(2, -.5), -std::pow(2, -.5) }};

  tensor_info cphase(double p) {
    return {{ 2, 2, 2, 2 }, { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, std::exp(1i * p)}};
  }

  std::vector<qtnh::uint> qft(qtnh::QTNHEnv& env, qtnh::TensorNetwork& tn, unsigned int nq, unsigned int dq) {
    using namespace qtnh;

    auto NQUBITS = nq;
    auto DQUBITS = dq;

    std::vector<qtnh::uint> con_ord(0);
    std::vector<qtnh::uint> qid(NQUBITS);
    std::vector<qtnh::tidx_tup_st> qidxi(NQUBITS);

    for (unsigned int i = 0; i < NQUBITS; ++i) {
      qid.at(i) = tn.createTensor<SDenseTensor>(env, plus_state.dims, plus_state.els);
      if (i < DQUBITS) {
        auto tid = tn.createTensor<ConvertTensor>(env, qtnh::tidx_tup{ 2 });
        auto bid = tn.createBond(qid.at(i), tid, {{ 0, 0 }}, true);
        con_ord.push_back(bid);

        qid.at(i) = tid;
        qidxi.at(i) = 1;
      } else {
        qidxi.at(i) = 0;
      }

      if (i > 0) {
        auto bid = tn.createBond(qid.at(i - 1), qid.at(i), {});
        con_ord.push_back(bid);
      }
    }

    for (int i = NQUBITS - 1; i >= 0; --i) {
      if (i < (int)DQUBITS) {
        int i1 = i, i2 = NQUBITS - 1;

        auto tid1 = tn.createTensor<SwapTensor>(env, 2, 2);
        auto tid2 = tn.createTensor<SDenseTensor>(env, hadamard.dims, hadamard.els);
        auto tid3 = tn.createTensor<SwapTensor>(env, 2, 2);

        auto bid1 = tn.createBond(qid.at(i1), tid1, {{ qidxi.at(i1), 0 }}, true);
        auto bid2 = tn.createBond(qid.at(i2), tid1, {{ qidxi.at(i2), 1 }}, true);
        auto bid3 = tn.createBond(tid1, tid2, {{ 3, 0 }});
        auto bid4 = tn.createBond(tid2, tid3, {{ 1, 1 }}, true);
        auto bid5 = tn.createBond(tid1, tid3, {{ 2, 0 }}, true);

        con_ord.push_back(bid1);
        con_ord.push_back(bid2);
        con_ord.push_back(bid3);
        con_ord.push_back(bid4);
        con_ord.push_back(bid5);

        qid.at(i1) = qid.at(i2) = tid3;
        qidxi.at(i1) = 2; qidxi.at(i2) = 3;
      } else {
        auto tid = tn.createTensor<SDenseTensor>(env, hadamard.dims, hadamard.els);
        auto bid = tn.createBond(qid.at(i), tid, {{ qidxi.at(i), 0 }});
        con_ord.push_back(bid);

        qid.at(i) = tid;
        qidxi.at(i) = 1;
      }

      for (int j = i - 1; j >= 0; --j) {
        auto ii = i, jj = j;
        if (i < (int)DQUBITS) {
          int i0 = i, i1 = NQUBITS - 1;

          auto tid = tn.createTensor<SwapTensor>(env, 2, 2);
          auto bid0 = tn.createBond(qid.at(i0), tid, {{ qidxi.at(i0), 0 }}, true);
          auto bid1 = tn.createBond(qid.at(i1), tid, {{ qidxi.at(i1), 1 }}, true);

          con_ord.push_back(bid0); con_ord.push_back(bid1);
          qid.at(i0) = qid.at(i1) = tid;
          qidxi.at(i0) = 2; qidxi.at(i1) = 3;
          ii = i1;
        }
        if (j < (int)DQUBITS) {
          int j0 = j, j1 = NQUBITS - 2;
          if (j1 == ii) j1 = NQUBITS - 1;

          auto tid = tn.createTensor<SwapTensor>(env, 2, 2);
          auto bid0 = tn.createBond(qid.at(j0), tid, {{ qidxi.at(j0), 0 }}, true);
          auto bid1 = tn.createBond(qid.at(j1), tid, {{ qidxi.at(j1), 1 }}, true);

          con_ord.push_back(bid0); con_ord.push_back(bid1);
          qid.at(j0) = qid.at(j1) = tid;
          qidxi.at(j0) = 2; qidxi.at(j1) = 3;
          jj = j1;
        }

        auto cp = cphase(M_PI / std::pow(2, j - i));
        auto tid = tn.createTensor<SDenseTensor>(env, cp.dims, cp.els);
        auto bid0 = tn.createBond(qid.at(ii), tid, {{ qidxi.at(ii), 0 }});
        auto bid1 = tn.createBond(qid.at(jj), tid, {{ qidxi.at(jj), 1 }});

        con_ord.push_back(bid0); con_ord.push_back(bid1);
        qid.at(ii) = qid.at(jj) = tid;
        qidxi.at(ii) = 2; qidxi.at(jj) = 3;

        if (i != ii) {
          auto tid = tn.createTensor<SwapTensor>(env, 2, 2);
          auto bid0 = tn.createBond(qid.at(i), tid, {{ qidxi.at(i), 0 }}, true);
          auto bid1 = tn.createBond(qid.at(ii), tid, {{ qidxi.at(ii), 1 }}, true);

          con_ord.push_back(bid0); con_ord.push_back(bid1);
          qid.at(i) = qid.at(ii) = tid;
          qidxi.at(i) = 2; qidxi.at(ii) = 3;
        }
        if (j != jj) {
          auto tid = tn.createTensor<SwapTensor>(env, 2, 2);
          auto bid0 = tn.createBond(qid.at(j), tid, {{ qidxi.at(j), 0 }}, true);
          auto bid1 = tn.createBond(qid.at(jj), tid, {{ qidxi.at(jj), 1 }}, true);

          con_ord.push_back(bid0); con_ord.push_back(bid1);
          qid.at(j) = qid.at(jj) = tid;
          qidxi.at(j) = 2; qidxi.at(jj) = 3;
        }
      }
    }

    for (unsigned int i = 0; i < NQUBITS / 2; ++i) {
      auto i0 = i, i1 = NQUBITS - i - 1;

      auto tid = tn.createTensor<SwapTensor>(env, 2, 2);
      auto bid0 = tn.createBond(qid.at(i0), tid, {{ qidxi.at(i0), 0 }}, true);
      auto bid1 = tn.createBond(qid.at(i1), tid, {{ qidxi.at(i1), 1 }}, true);

      con_ord.push_back(bid0); con_ord.push_back(bid1);
      qid.at(i0) = qid.at(i1) = tid;
      qidxi.at(i0) = 2; qidxi.at(i1) = 3;
    }

    return con_ord;
  }
}

#endif