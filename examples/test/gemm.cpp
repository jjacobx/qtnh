#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;
using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  // Set up tensors. 
  auto dims_a = tidx_tup { 2, 2, 2, 2, 2, 2 };
  auto els_a = std::vector<tel>(utils::dims_to_size(dims_a), 0);
  std::iota(els_a.begin(), els_a.end(), 0);
  
  tptr tp_a = DenseTensor::make(env, {}, dims_a, std::move(els_a));
  tp_a = Tensor::rescatter(std::move(tp_a), 3);

  auto dims_b = tidx_tup { 2, 2, 2, 2, 2, 2 };
  auto els_b = std::vector<tel>(utils::dims_to_size(dims_b), 0);
  std::iota(els_b.begin(), els_b.end(), 0);

  tptr tp_b = DenseTensor::make(env, {}, dims_b, std::move(els_b));
  tp_b = Tensor::rescatter(std::move(tp_b), 3);

  std::cout << "P" << env.proc_id << ": A = " << *tp_a << "\n";
  std::cout << "P" << env.proc_id << ": B = " << *tp_b << "\n";

  // Permute tensors into matrix-like form. 
  IndexGroup ig_a(
    { "rc", "rd", "rb", "cc", "cd", "cb" }, 
    {{}, { 0, 1 }, { 2 }, {}, { 3 }, { 4, 5 }}
  );
  IndexGroup ig_b(
    { "rc", "rd", "rb", "cc", "cd", "cb" }, 
    {{}, { 0, 1 }, { 2 }, {}, { 3 }, { 4, 5 }}
  );

  ig_a.reorder({ "rd", "cd", "cc", "cb", "rc", "rb" });
  ig_b.reorder({ "rd", "cd", "cc", "cb", "rc", "rb" });

  tp_a = Tensor::permute(std::move(tp_a), ig_a.ptup().toTar().tup());
  tp_b = Tensor::permute(std::move(tp_b), ig_b.ptup().toTar().tup());

  // Convert tensors to BC matrices and multiply. 
  using namespace lalg;
  ProcGrid pg(4, 2);

  BlockCyclicMatrix a(pg, { 8, 8 }, { 2, 4 }, tp_a->cast<DenseTensor>()->extractEls());
  BlockCyclicMatrix b(pg, { 8, 8 }, { 2, 4 }, tp_a->cast<DenseTensor>()->extractEls());

  auto c = PZGEMM(std::move(a), std::move(b), false, false);

  // Convert output BC matrix back to result. 
  auto dis_dims_c = tidx_tup { 2, 2, 2 };
  auto loc_dims_c = tidx_tup { 2, 2, 2 };
  tptr tp_c = DenseTensor::make(env, dis_dims_c, loc_dims_c, c.extractEls());

  IndexGroup ig_c(
    { "rc", "rd", "rb", "cc", "cd", "cb" }, 
    {{}, { 0, 1 }, { 2 }, {}, { 3 }, { 4, 5 }}
  );
  ig_c.reorder({ "rd", "cd", "cc", "cb", "rc", "rb" });

  tp_c = Tensor::permute(std::move(tp_c), ig_c.ptup().inv().toTar().tup());

  std::cout << "P" << env.proc_id << ": C = " << *tp_c << "\n";

  utils::barrier();

  BlockCyclicMatrix a2(pg, { 2, 2 }, { 1, 1 }, { 1, 1 }, { 1, 2, 3, 4 });
  BlockCyclicMatrix b2(pg, { 2, 2 }, { 1, 1 }, { 1, 1 }, { 1, 2, 3, 4 });

  auto c2 = PZGEMM(std::move(a2), std::move(b2));
  auto els_c2 = c2.extractEls();
  
  std::cout << "P" << env.proc_id << " | els_c2 = ";
  for (auto e : els_c2) {
    std::cout << e << ", ";
  }
  std::cout << "\n";
}
