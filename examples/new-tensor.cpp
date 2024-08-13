#include <iostream>

#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;

using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  std::vector<tel> els1 = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  std::vector<tel> els2 = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

  tptr tp1 = DenseTensor::make(env, {}, { 2, 2, 2 }, std::vector<tel>(els1));
  tptr tp2 = DenseTensor::make(env, {}, { 4, 2 }, std::vector<tel>(els2));

  if (tp1->bc().active) {
    std::cout << env.proc_id << " | T1[0] = " << (*tp1)[0] << "\n";
  }
  if (tp2->bc().active) {
    std::cout << env.proc_id << " | T2[0] = " << (*tp2)[0] << "\n";
  }
  
  tp1 = Tensor::rescatter(std::move(tp1), 1);
  if (tp1->bc().active) {
    std::cout << env.proc_id << " | T1[0] (scatter 1) = " << (*tp1)[0] << "\n";
  }

  tptr tp3 = Tensor::contract(std::move(tp1), std::move(tp2), {{ 1, 1 }});
  if (tp3->bc().active) {
    std::cout << env.proc_id << " | T3[0] = " << (*tp3)[0] << "\n";
  }

  tp1 = DenseTensor::make(env, {}, { 2, 2, 2 }, std::vector<tel>(els1));
  tp1 = Tensor::rescatter(std::move(tp1), 2);
  if (tp1->bc().active) {
    std::cout << env.proc_id << " | T1[0] (scatter 2) = " << (*tp1)[0] << "\n";
  }

  tp1 = Tensor::rescatter(std::move(tp1), -2);
  if (tp1->bc().active) {
    std::cout << env.proc_id << " | T1[0] (gather 2) = " << (*tp1)[0] << "\n";
  }

  tp1 = DenseTensor::make(env, {}, { 2, 2, 2 }, std::vector<tel>(els1));
  tp2 = DenseTensor::make(env, {}, { 4, 2 }, std::vector<tel>(els2));

  tp3 = Tensor::contract(std::move(tp1), std::move(tp2), {});
  if (tp3->has({1, 1, 1, 3, 1})) {
    std::cout << env.proc_id << ": T3[(1, 1, 1, 3, 1)] (tensor product) = " << tp3->at({1, 1, 1, 3, 1}) << "\n";
  }

  return 0;
}
