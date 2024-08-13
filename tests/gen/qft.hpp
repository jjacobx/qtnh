#ifndef QFT_TENSORS_HPP 
#define QFT_TENSORS_HPP

#include "validation-primitives.hpp"
#include "tensor/network.hpp"

using namespace std::complex_literals;

namespace gen {
  tensor_info zero_state {{ 2 }, { 1, 0 }};
  tensor_info plus_state {{ 2 }, { std::pow(2, -.5), std::pow(2, -.5) }};
  tensor_info hadamard {{ 2, 2 }, { std::pow(2, -.5),  std::pow(2, -.5), std::pow(2, -.5), -std::pow(2, -.5) }};

  tensor_info cphase(double p) {
    return {{ 2, 2, 2, 2 }, { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, std::exp(1i * p)}};
  }

  std::vector<qtnh::uint> qft(qtnh::QTNHEnv& env, qtnh::TensorNetwork& tn, qtnh::uint nq, qtnh::uint dq) {
    using namespace qtnh;

    auto NQUBITS = nq;
    auto DQUBITS = dq;

    std::vector<uint> con_ord(0);
    std::vector<uint> qid(NQUBITS);
    std::vector<tidx_tup_st> qidxi(NQUBITS);

    for (uint i = 0; i < NQUBITS; ++i) {
      qid.at(i) = tn.make<DenseTensor>(env, tidx_tup {}, plus_state.dims, std::vector<tel>(plus_state.els));
      qidxi.at(i) = 0;

      if (i < DQUBITS) {
        auto tid = tn.make<IdenTensor>(env, tidx_tup { 2 }, tidx_tup { 2 }, 0, 0);
        auto bid = tn.addBond(qid.at(i), tid, {{ 0, 1 }});
        con_ord.push_back(bid);

        qid.at(i) = tid;
        qidxi.at(i) = 0;
      }

      if (i > 0) {
        auto bid = tn.addBond(qid.at(i - 1), qid.at(i), {});
        con_ord.push_back(bid);
      }
    }

    for (uint i = 0; i < NQUBITS; ++i) {
      uint hid = tn.make<SymmTensor>(env, tidx_tup {}, hadamard.dims, 0, std::vector<tel>(hadamard.els));
      auto tid1 = hid, tid2 = hid;
      uint idxi2 = 1;

      // Distribute the Hadamard gate if it acts on a distributed qubit. 
      if (i < DQUBITS) {
        tid1 = tn.make<IdenTensor>(env, tidx_tup { 2 }, tidx_tup { 2 }, 1, 0);
        auto bid1 = tn.addBond(tid1, hid, {{ 1, 0 }});
        con_ord.push_back(bid1);

        tid2 = tn.make<IdenTensor>(env, tidx_tup { 2 }, tidx_tup { 2 }, 0, 0);
        auto bid2 = tn.addBond(hid, tid2, {{ 1, 1 }});
        con_ord.push_back(bid2);
        idxi2 = 0;
      }

      auto bid = tn.addBond(qid.at(i), tid1, {{ qidxi.at(i), 0 }});
      con_ord.push_back(bid);

      qid.at(i) = tid2;
      qidxi.at(i) = idxi2;

      for (uint j = i + 1; j < NQUBITS; ++j) {
        auto cp = cphase(M_PI / std::pow(2, j - i));
        uint cpid = tn.make<SymmTensor>(env, tidx_tup {}, cp.dims, 0, std::vector<tel>(cp.els));
        auto tid1 = cpid, tid2 = cpid, tid3 = cpid, tid4 = cpid;
        auto idxi1 = 0, idxi2 = 1, idxi3 = 2, idxi4 = 3;
        
        if (i < DQUBITS) {
          tid1 = tn.make<IdenTensor>(env, tidx_tup { 2 }, tidx_tup { 2 }, 1, 0);
          auto bid1 = tn.addBond(tid1, cpid, {{ 1, 0 }});
          con_ord.push_back(bid1);
          idxi1 = 0;

          tid3 = tn.make<IdenTensor>(env, tidx_tup { 2 }, tidx_tup { 2 }, 0, 0);
          auto bid2 = tn.addBond(cpid, tid3, {{ 2, 1 }});
          con_ord.push_back(bid2);
          idxi3 = 0;
        }

        if (j < DQUBITS) {
          tid2 = tn.make<IdenTensor>(env, tidx_tup { 2 }, tidx_tup { 2 }, 1, 0);
          auto bid1 = tn.addBond(tid2, cpid, {{ 1, 1 }});
          con_ord.push_back(bid1);
          idxi2 = 0;

          tid4 = tn.make<IdenTensor>(env, tidx_tup { 2 }, tidx_tup { 2 }, 0, 0);
          auto bid2 = tn.addBond(cpid, tid4, {{ 3, 1 }});
          con_ord.push_back(bid2);
          idxi4 = 0;
        }

        auto bid1 = tn.addBond(qid.at(i), tid1, {{ qidxi.at(i), idxi1 }});
        auto bid2 = tn.addBond(qid.at(j), tid2, {{ qidxi.at(j), idxi2 }});
        con_ord.push_back(bid1);
        con_ord.push_back(bid2);

        qid.at(i) = tid3; qidxi.at(i) = idxi3;
        qid.at(j) = tid4; qidxi.at(j) = idxi4;
      }
    }

    return con_ord;
  }
}

#endif