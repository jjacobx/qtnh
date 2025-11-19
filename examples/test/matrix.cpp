#include <cmath>
#include <iostream>
#include "qtnh.hpp"
#include "blas/routines.hpp"

using namespace qtnh;
using namespace qtnh::lalg;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  std::vector<tel> els(4, { 0.0, 0.0 });
  if (env.proc_id == 0) {
    els.at(0) = { 1.0, 0.0 };
  } else if (env.proc_id == 1) {
    els.at(0) = { 2.0, 0.0 };
  } else if (env.proc_id == 2) {
    els.at(3) = { 3.0, 0.0 };
  } else if (env.proc_id == 3) {
    els.at(3) = { 4.0, 0.0 };
  }
  // if (env.proc_id == 0) {
  //   els.at(0) = { 1.0, 0.0 };
  //   els.at(3) = { 3.0, 0.0 };
  // } else if (env.proc_id == 1) {

  // } else if (env.proc_id == 2) {

  // } else if (env.proc_id == 3) {
  //   els.at(0) = { 2.0, 0.0 };
  //   els.at(3) = { 4.0, 0.0 };
  // }

  ProcGrid pg(2, 2);
  // BlockCyclicMatrix m(pg, { 2, 2 }, { 2, 2 }, { 1, 1 }, std::move(els));
  // BlockCyclicMatrix m2(pg, { 2, 2 }, { 2, 2 }, { 1, 1 }, m.copyEls());

  BlockCyclicMatrix m(pg, { 1, 1 }, { 2, 2 }, { 2, 2 }, std::move(els));
  BlockCyclicMatrix m2(pg, { 1, 1 }, { 2, 2 }, { 2, 2 }, m.copyEls());

  PZLAPRNT(m, "M");
  if (utils::is_root()) {
    std::cout << m.desc9() << std::endl;
  }

  utils::barrier();
  

  auto [q, r, pt] = PZGEQPD(std::move(m));

  PZLAPRNT(q, "Q");
  PZLAPRNT(r, "R");
  PZLAPRNT(pt, "Pt");

  auto qr = PZGEMM(std::move(q), std::move(r));
  auto qrpt = PZGEMM(std::move(qr), std::move(pt));

  PZLAPRNT(m2, "M");
  PZLAPRNT(qrpt, "QRPt");
}
