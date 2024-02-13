#include <iostream>

#include "tensor-network.hpp"
#include "indexing.hpp"

using namespace std::complex_literals;
using namespace qtnh::ops;

int main() {
  qtnh::QTNHEnv my_env;

  qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  qtnh::tidx_tup t2_dims = { 4, 2 };

  std::vector<qtnh::tel> t1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  std::vector<qtnh::tel> t2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

  qtnh::SDenseTensor t1(my_env, t1_dims, t1_els);
  qtnh::SDenseTensor t2(my_env, t2_dims, t2_els);
  auto t3 = t1.distribute(1);

  std::vector<qtnh::wire> wires1(1, {1, 2});
  qtnh::Bond b1({t2.getID(), t3.getID()}, wires1);

  qtnh::TensorNetwork tn;
  tn.insertTensor(t2);
  tn.insertTensor(t3);
  tn.insertBond(b1);

  auto t4id = tn.contractBond(b1.getID());
  auto& t_out = tn.getTensor(t4id);

  std::cout << my_env.proc_id << " | Tout = " << t_out << std::endl;
  t_out.swap(0, 2);
  std::cout << my_env.proc_id << " | Tout_s1 = " << t_out << std::endl;


  return 0;
}
