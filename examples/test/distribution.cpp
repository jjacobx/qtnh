#include <iostream>
#include "qtnh.hpp"

using namespace qtnh;
using namespace std::complex_literals;

int main(int argc, char* argv[]) {
  using namespace qtnh;

  QTNHEnv env;
  TensorNetwork tn;

  tidx_tup t1_loc_dims(10, 2);
  std::vector<tel> t1_els(1024);
  std::iota(t1_els.begin(), t1_els.end(), 0);

  tptr tp1 = DenseTensor::make(env, {}, t1_loc_dims, std::move(t1_els));
  std::cout << env.proc_id << " | T1 = " << *tp1 << std::endl;

  utils::barrier();
  if (utils::is_root()) std::cout << "\n\n=======================================================\n\n";
  utils::barrier();

  tp1 = Tensor::rescatter(std::move(tp1), 6);
  std::cout << env.proc_id << " | T1 = " << *tp1 << std::endl;

    utils::barrier();
  if (utils::is_root()) std::cout << "\n\n=======================================================\n\n";
  utils::barrier();

  tptr tp2 = DenseTensor::make(env, {}, { 2, 2, 2, 2 }, { 
    1, 0, 0, 0, 
    0, 1, 0, 0, 
    0, 0, 1, 0, 
    0, 0, 0, 1 
  });
  tp2 = Tensor::rescatter(std::move(tp2), 4);

  auto params = ConParams({{ 4, 0 }, { 5, 1 }});
  tp1 = pcon(std::move(tp1), tp2->copy(), params).contract();
  std::cout << env.proc_id << " | T1 = " << *tp1 << std::endl;

  utils::barrier();
  if (utils::is_root()) std::cout << "\n\n=======================================================\n\n";
  utils::barrier();

  std::cout << env.proc_id << "/" << env.num_processes << "\n";
}
