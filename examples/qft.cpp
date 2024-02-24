#include <cmath>
#include <complex>
#include <iostream>

#include "qtnh.hpp"

using namespace std::complex_literals;

using namespace qtnh;
using namespace qtnh::ops;

const int NQUBITS = 4; 

Tensor* Q0(const QTNHEnv& env) {
  auto* q0 = new SDenseTensor(env, { 2 }, { 1.0, 0.0 });
  return q0;
}

Tensor* H(const QTNHEnv& env) {
  std::vector<qtnh::tel> els({ 
    std::pow(2, -.5),  std::pow(2, -.5), 
    std::pow(2, -.5), -std::pow(2, -.5) 
  });

  auto* h = new SDenseTensor(env, { 2, 2 }, els);
  return h;
}

Tensor* CP(const QTNHEnv& env, double p) {
  std::vector<qtnh::tel> els({ 
    1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 1, 0, 
    0, 0, 0, std::exp(1i * p)
  });

  auto* cp = new SDenseTensor(env, { 2, 2, 2, 2 }, els);
  return cp;
}

int main() {
  QTNHEnv env;
  TensorNetwork tn;

  std::vector<qtnh::uint> con_ord(0);
  std::vector<qtnh::uint> lastq(NQUBITS);
  std::vector<qtnh::tidx_tup_st> dimq(NQUBITS);

  for (auto i = 0; i < NQUBITS; ++i) {
    lastq.at(i) = tn.insertTensor(*Q0(env));
    dimq.at(i) = 0;

    if (i > 0) {
      auto* b = new Bond({ lastq.at(i), lastq.at(i - 1) }, {});
      con_ord.push_back(tn.insertBond(*b));
    }
  }

  for (auto i = 0; i < NQUBITS; ++i) {
    auto idh = tn.insertTensor(*H(env));
    auto* bh = new Bond({ lastq.at(i), idh }, {{ dimq.at(i), 0 }});
    con_ord.push_back(tn.insertBond(*bh));

    lastq.at(i) = idh;
    dimq.at(i) = 1;

    for (auto j = i + 1; j < NQUBITS; ++j) {
      auto idcp = tn.insertTensor(*CP(env, M_PI / std::pow(2, j - i)));

      auto* bcp1 = new Bond({ lastq.at(i), idcp }, {{ dimq.at(i), 0 }});
      auto* bcp2 = new Bond({ lastq.at(j), idcp }, {{ dimq.at(j), 1 }});
      con_ord.push_back(tn.insertBond(*bcp1));
      con_ord.push_back(tn.insertBond(*bcp2));

      lastq.at(i) = idcp; 
      lastq.at(j) = idcp; 
      dimq.at(i) = 2;
      dimq.at(j) = 3;
    }
  }

  tn.print();

  auto id = tn.contractAll(con_ord);
  std::cout << tn.getTensor(id) << std::endl;

  return 0;
}