#include <chrono>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::chrono;
using namespace std::complex_literals;

uint Q0(const QTNHEnv& env, TensorNetwork& tn) {
  std::vector<tel> els = { 1.0, 0.0 };
  return tn.make<DenseTensor>(env, tidx_tup {}, tidx_tup { 2 }, std::move(els));
}

qtnh::uint Q1(const QTNHEnv& env, TensorNetwork& tn) {
  std::vector<tel> els = { 0.0, 1.0 };
  return tn.make<DenseTensor>(env, tidx_tup {}, tidx_tup { 2 }, std::move(els));
}

qtnh::uint Qp(const QTNHEnv& env, TensorNetwork& tn) {
  std::vector<tel> els = { std::pow(2, -.5), std::pow(2, -.5) };
  return tn.make<DenseTensor>(env, tidx_tup {}, tidx_tup { 2 }, std::move(els));
}

qtnh::uint RESC(const QTNHEnv& env, TensorNetwork& tn) {
  return tn.make<RescTensor>(env, 2);
}

qtnh::uint H(const QTNHEnv& env, TensorNetwork& tn) {
  std::vector<tel> els = { 
    std::pow(2, -.5),  std::pow(2, -.5), 
    std::pow(2, -.5), -std::pow(2, -.5) 
  };

  return tn.make<SymmTensor>(env, tidx_tup {}, tidx_tup { 2, 2 }, std::move(els));
}

qtnh::uint CPH(const QTNHEnv& env, TensorNetwork& tn, double p) {
  qtnh::tidx_tup dims{ 2, 2, 2, 2 };
  std::vector<qtnh::tel> els = { 
    1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 1, 0, 
    0, 0, 0, std::exp(1i * p)
  };

  return tn.make<SymmTensor>(env, tidx_tup {}, tidx_tup { 2, 2, 2, 2 }, std::move(els));
}

int main(int argc, char* argv[]) {
  using namespace qtnh;

  auto DQ = 5U;
  auto LQ = 2U;

  if (argc > 2) {
    DQ = static_cast<unsigned int>(strtol(argv[1], nullptr, 0));
    LQ = static_cast<unsigned int>(strtol(argv[2], nullptr, 0));
  }

  auto NQ = DQ + LQ;

  QTNHEnv env;
  TensorNetwork tn;

  std::vector<uint> con_ord(0);
  std::vector<uint> qid(NQ);
  std::vector<tidx_tup_st> qidxi(NQ);

  for (uint i = 0; i < NQ; ++i) {
    qid.at(i) = Qp(env, tn);
    qidxi.at(i) = 0;

    if (i < DQ) {
      auto tid = RESC(env, tn);
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

  for (uint i = 0; i < NQ; ++i) {
    uint hid = H(env, tn);
    auto tid1 = hid, tid2 = hid;
    uint idxi2 = 1;

    // Distribute the Hadamard gate if it acts on a distributed qubit. 
    if (i < DQ) {
      tid1 = RESC(env, tn);
      auto bid1 = tn.addBond(tid1, hid, {{ 1, 0 }});
      con_ord.push_back(bid1);

      tid2 = RESC(env, tn);
      auto bid2 = tn.addBond(hid, tid2, {{ 1, 1 }});
      con_ord.push_back(bid2);
      idxi2 = 0;
    }

    auto bid = tn.addBond(qid.at(i), tid1, {{ qidxi.at(i), 0 }});
    con_ord.push_back(bid);

    qid.at(i) = tid2;
    qidxi.at(i) = idxi2;

    for (uint j = i + 1; j < NQ; ++j) {
      auto cpid = CPH(env, tn, M_PI / std::pow(2, j - i));
      auto tid1 = cpid, tid2 = cpid, tid3 = cpid, tid4 = cpid;
      auto idxi1 = 0, idxi2 = 1, idxi3 = 2, idxi4 = 3;
      
      if (i < DQ) {
        tid1 = RESC(env, tn);
        auto bid1 = tn.addBond(tid1, cpid, {{ 1, 0 }});
        con_ord.push_back(bid1);
        idxi1 = 0;

        tid3 = RESC(env, tn);
        auto bid2 = tn.addBond(cpid, tid3, {{ 2, 1 }});
        con_ord.push_back(bid2);
        idxi3 = 0;
      }

      if (j < DQ) {
        tid2 = RESC(env, tn);
        auto bid1 = tn.addBond(tid2, cpid, {{ 1, 1 }});
        con_ord.push_back(bid1);
        idxi2 = 0;

        tid4 = RESC(env, tn);
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

  if (utils::is_root()) std::cout << "Starting contraction\n";

  utils::barrier();
  auto start = high_resolution_clock::now();

  auto tid = tn.contractAll(con_ord);

  utils::barrier();
  auto stop = high_resolution_clock::now();

  auto tp = tn.extract(tid);
  auto delta = duration_cast<milliseconds>(stop - start);

  if (utils::is_root()) {
    std::cout << "\n";
    std::cout << "T[0] = " << (*tp)[0] << "\n";
    std::cout << "Time taken: " << delta.count() << " ms\n";
  }

  return 0;
}
