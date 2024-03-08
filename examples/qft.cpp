#include <cmath>
#include <complex>
#include <iostream>

#include "qtnh.hpp"

using namespace std::complex_literals;

using namespace qtnh;
using namespace qtnh::ops;

const unsigned int NQUBITS = 4; 
const unsigned int DQUBITS = 2;

qtnh::uint Q0(const QTNHEnv& env, TensorNetwork& tn) {
  qtnh::tidx_tup dims{ 2 };
  std::vector<qtnh::tel> els{ 1.0, 0.0 };
  return tn.createTensor<SDenseTensor>(env, dims, els);
}

qtnh::uint Q1(const QTNHEnv& env, TensorNetwork& tn) {
  qtnh::tidx_tup dims{ 2 };
  std::vector<qtnh::tel> els{ 0.0, 1.0 };
  return tn.createTensor<SDenseTensor>(env, dims, els);
}

qtnh::uint Qp(const QTNHEnv& env, TensorNetwork& tn) {
  qtnh::tidx_tup dims{ 2 };
  std::vector<qtnh::tel> els{ std::pow(2, -.5), std::pow(2, -.5) };
  return tn.createTensor<SDenseTensor>(env, dims, els);
}

qtnh::uint DIST(const QTNHEnv& env, TensorNetwork& tn) {
  return tn.createTensor<ConvertTensor>(env, qtnh::tidx_tup{ 2 });
}

qtnh::uint SWAP(const QTNHEnv& env, TensorNetwork& tn) {
  return tn.createTensor<SwapTensor>(env, 2, 2);
}

qtnh::uint H(const QTNHEnv& env, TensorNetwork& tn) {
  qtnh::tidx_tup dims{ 2, 2 };
  std::vector<qtnh::tel> els({ 
    std::pow(2, -.5),  std::pow(2, -.5), 
    std::pow(2, -.5), -std::pow(2, -.5) 
  });

  return tn.createTensor<SDenseTensor>(env, dims, els);
}

qtnh::uint CP(const QTNHEnv& env, TensorNetwork& tn, double p) {
  qtnh::tidx_tup dims{ 2, 2, 2, 2 };
  std::vector<qtnh::tel> els({ 
    1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 1, 0, 
    0, 0, 0, std::exp(1i * p)
  });

  return tn.createTensor<SDenseTensor>(env, dims, els);
}

int main() {
  QTNHEnv env;
  TensorNetwork tn;

  std::vector<qtnh::uint> con_ord(0);
  std::vector<qtnh::uint> qid(NQUBITS);
  std::vector<qtnh::tidx_tup_st> qidxi(NQUBITS);

  for (unsigned int i = 0; i < NQUBITS; ++i) {
    qid.at(i) = Qp(env, tn);
    if (i < DQUBITS) {
      auto tid = DIST(env, tn);
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

      auto tid1 = SWAP(env, tn);
      auto tid2 = H(env, tn);
      auto tid3 = SWAP(env, tn);

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
      auto tid = H(env, tn);
      auto bid = tn.createBond(qid.at(i), tid, {{ qidxi.at(i), 0 }});
      con_ord.push_back(bid);

      qid.at(i) = tid;
      qidxi.at(i) = 1;
    }

    for (int j = i - 1; j >= 0; --j) {
      auto ii = i, jj = j;
      if (i < (int)DQUBITS) {
        int i0 = i, i1 = NQUBITS - 1;

        auto tid = SWAP(env, tn);
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

        auto tid = SWAP(env, tn);
        auto bid0 = tn.createBond(qid.at(j0), tid, {{ qidxi.at(j0), 0 }}, true);
        auto bid1 = tn.createBond(qid.at(j1), tid, {{ qidxi.at(j1), 1 }}, true);

        con_ord.push_back(bid0); con_ord.push_back(bid1);
        qid.at(j0) = qid.at(j1) = tid;
        qidxi.at(j0) = 2; qidxi.at(j1) = 3;
        jj = j1;
      }

      auto tid = CP(env, tn, M_PI / std::pow(2, j - i));
      auto bid0 = tn.createBond(qid.at(ii), tid, {{ qidxi.at(ii), 0 }});
      auto bid1 = tn.createBond(qid.at(jj), tid, {{ qidxi.at(jj), 1 }});

      con_ord.push_back(bid0); con_ord.push_back(bid1);
      qid.at(ii) = qid.at(jj) = tid;
      qidxi.at(ii) = 2; qidxi.at(jj) = 3;

      if (i != ii) {
        auto tid = SWAP(env, tn);
        auto bid0 = tn.createBond(qid.at(i), tid, {{ qidxi.at(i), 0 }}, true);
        auto bid1 = tn.createBond(qid.at(ii), tid, {{ qidxi.at(ii), 1 }}, true);

        con_ord.push_back(bid0); con_ord.push_back(bid1);
        qid.at(i) = qid.at(ii) = tid;
        qidxi.at(i) = 2; qidxi.at(ii) = 3;
      }
      if (j != jj) {
        auto tid = SWAP(env, tn);
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

    auto tid = SWAP(env, tn);
    auto bid0 = tn.createBond(qid.at(i0), tid, {{ qidxi.at(i0), 0 }}, true);
    auto bid1 = tn.createBond(qid.at(i1), tid, {{ qidxi.at(i1), 1 }}, true);

    con_ord.push_back(bid0); con_ord.push_back(bid1);
    qid.at(i0) = qid.at(i1) = tid;
    qidxi.at(i0) = 2; qidxi.at(i1) = 3;
  }

  if (env.proc_id == 0) tn.print();

  // Contract the network and extract result. 
  auto id = tn.contractAll(con_ord);
  auto tfu = tn.extractTensor(id);
  // tn should be empty now. 

  MPI_Barrier(MPI_COMM_WORLD);
  std::cout << env.proc_id << " | " << *tfu << "\n";

  return 0;
}