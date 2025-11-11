#include <cmath>
#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::lalg;
using namespace std::complex_literals;

// int main() {
//   QTNHEnv env;

//   std::vector<tel> els(8, { 0.0, 0.0 });
//   if (env.proc_id == 0) {
//     els.at(0) = { 1.0 / std::sqrt(2), 0.0 };
//   } else if (env.proc_id == 1) {
//     els.at(0) = { -1.0 / std::sqrt(2), 0.0 };
//   } else if (env.proc_id == 2) {
//     els.at(5) = { 0.0, -1.0 / std::sqrt(2) };
//   } else if (env.proc_id == 3) {
//     els.at(5) = std::exp(tel { 0.0, 2.0 * M_PI / 3.0 }) / std::sqrt(2);
//   }

//   ProcGrid pg(2, 2, 0);
//   BlockCyclicMatrix m(pg, { 2, 2 }, { 2, 2 }, { 1, 2 }, std::move(els));
//   BlockCyclicMatrix m2(pg, { 2, 2 }, { 2, 2 }, { 1, 2 }, m.copyEls());

//   PZLAPRNT(m, "M");

//   auto [q, r, p] = PZGEQPD(std::move(m));

//   PZLAPRNT(q, "Q");
//   PZLAPRNT(r, "R");
//   PZLAPRNT(p, "P");

//   auto qr = PZGEMM(std::move(q), std::move(r));
//   // auto qrp = PZGEMM(std::move(qr), std::move(p));

//   PZLAPRNT(qr, "QR");

//   auto mp = PZGEMM(std::move(m2), std::move(p));

//   PZLAPRNT(mp, "MP");
// }

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

  ProcGrid pg(2, 2, 0);
  BlockCyclicMatrix m(pg, { 1, 1 }, { 2, 2 }, { 2, 2 }, std::move(els));
  BlockCyclicMatrix m2(pg, { 1, 1 }, { 2, 2 }, { 2, 2 }, m.copyEls());

  PZLAPRNT(m, "M");

  auto [q, r, p] = PZGEQPD(std::move(m));

  PZLAPRNT(q, "Q");
  PZLAPRNT(r, "R");
  PZLAPRNT(p, "P");

  auto qr = PZGEMM(std::move(q), std::move(r));
  // auto qrp = PZGEMM(std::move(qr), std::move(p));

  PZLAPRNT(qr, "QR");

  auto mp = PZGEMM(std::move(m2), std::move(p));

  PZLAPRNT(mp, "MP");
}

// int main2() {
//   QTNHEnv env;

//   std::vector<tel> els(32, { 0.0, 0.0 });
//   if (env.proc_id == 0) {
//     els.at(0) = { 1.0 / std::sqrt(2), 0.0 };
//     els.at(9) = { -1.0 / std::sqrt(2), 0.0 };
//     els.at(20) = { 0.0, -1.0 / std::sqrt(2) };
//     els.at(28) = std::exp(tel { 0.0, 2.0 * M_PI / 3.0 }) / std::sqrt(2);
//   }

//   ProcGrid pg(1, 1, 0);
//   BlockCyclicMatrix m(pg, { 1, 2 }, { 1, 1 }, { 4, 4 }, std::move(els));
//   BlockCyclicMatrix m2(pg, { 1, 2 }, { 1, 1 }, { 4, 4 }, m.copyEls());

//   PZLAPRNT(m, "M");

//   auto [q, r, p] = PZGEQPD(std::move(m));

//   PZLAPRNT(q, "Q");
//   PZLAPRNT(r, "R");
//   PZLAPRNT(p, "P");

//   auto qr = PZGEMM(std::move(q), std::move(r));
//   // auto qrp = PZGEMM(std::move(qr), std::move(p));

//   PZLAPRNT(qr, "QR");

//   auto mp = PZGEMM(std::move(m2), std::move(p));

//   PZLAPRNT(mp, "MP");
// }