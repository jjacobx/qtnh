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

  if (tp1->bc().isActive()) {
    std::cout << env.proc_id << " | T1[0] = " << (*tp1)[0] << "\n";
  }
  if (tp2->bc().isActive()) {
    std::cout << env.proc_id << " | T2[0] = " << (*tp2)[0] << "\n";
  }
  
  tp1 = Tensor::rescatter(std::move(tp1), 1);
  if (tp1->bc().isActive()) {
    std::cout << env.proc_id << " | T1[0] (scatter 1) = " << (*tp1)[0] << "\n";
  }

  auto params = ConParams({{ 1, 1 }});
  tptr tp3 = pcon(std::move(tp1), std::move(tp2), params).contract();
  if (tp3->bc().isActive()) {
    std::cout << env.proc_id << " | T3[0] = " << (*tp3)[0] << "\n";
  }

  tp1 = DenseTensor::make(env, {}, { 2, 2, 2 }, std::vector<tel>(els1));
  tp1 = Tensor::rescatter(std::move(tp1), 2);
  if (tp1->bc().isActive()) {
    std::cout << env.proc_id << " | T1[0] (scatter 2) = " << (*tp1)[0] << "\n";
  }

  tp1 = Tensor::rescatter(std::move(tp1), -2);
  if (tp1->bc().isActive()) {
    std::cout << env.proc_id << " | T1[0] (gather 2) = " << (*tp1)[0] << "\n";
  }

  using dpcon = PairContractor<DenseTensor, DenseTensor>;
  auto tp_d1 = DenseTensor::make(env, {}, { 2, 4 }, std::vector<tel>(els1));
  tp_d1 = Tensor::cast<DenseTensor>(Tensor::rescatter(std::move(tp_d1), 1));
  auto tp_d2 = DenseTensor::make(env, {}, { 2, 2, 2 }, std::vector<tel>(els2));
  tp_d2 = Tensor::cast<DenseTensor>(Tensor::rescatter(std::move(tp_d2), 2));

  params = ConParams(std::vector<wire> {});
  auto tp_d3 = dpcon(Tensor::cast<DenseTensor>(tp_d1->copy()), 
                     Tensor::cast<DenseTensor>(tp_d2->copy()), params).contract();

  if (tp_d3->has({ 1, 1, 1, 3, 1 })) {
    std::cout << env.proc_id << ": T3[(1, 1, 1, 3, 1)] (tensor product) = " << tp_d3->at({ 1, 1, 1, 3, 1 }) << "\n";
  }

  return 0;
}
