#include <iostream>

#include "dense-tensor.hpp"

using namespace std::complex_literals;

int main() {
  qtnh::QTNHEnv my_env;

  std::vector<qtnh::tel> t1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  qtnh::tidx_tup t1_dims = { 2, 2, 2 };

  qtnh::SDenseTensor t1(my_env, t1_dims, t1_els);

  qtnh::tidx_tup idxs1 = { 0, 1, 0 };
  std::cout << my_env.proc_id << ": t1[0, 1, 0] = " << t1.getLocEl(idxs1).value() << std::endl;
  std::cout << my_env.proc_id << ": t1.id = " << t1.getID() << std::endl;

  auto t2 = t1.distribute(1);

  qtnh::tidx_tup idxs2 = { 0, 1 };
  std::cout << my_env.proc_id << ": t2[r, 0, 1] = " << t2.getLocEl(idxs2).value() << std::endl;
  std::cout << my_env.proc_id << ": t2.id = " << t2.getID() << std::endl;

  return 0;
}
