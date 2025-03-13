#include <iostream>

#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;

using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  tptr tp1 = DenseTensor::make(env, {}, { 2, 2 }, { 1, 0, 0, 1 });
  tptr tp2 = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 });

  pcon con(std::move(tp1), std::move(tp2), ConParams(std::vector<wire> {{ 0, 2 }}));
  tptr tp3 = con.contract();

  std::cout << *tp3 << "\n";
}
