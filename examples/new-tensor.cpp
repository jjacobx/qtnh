#include <iostream>

#include "dense-tensor.hpp"

using namespace std::complex_literals;

int main() {
  qtnh::QTNHEnv my_env;

  std::vector<qtnh::tel> dt1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  qtnh::tidx_tup dt1_dims = { 2, 2, 2 };

  qtnh::SDenseTensor dt1(my_env, dt1_dims, dt1_els);

  qtnh::tidx_tup idxs1 = { 0, 1, 0 };
  std::cout << my_env.proc_id << ": t1[0, 1, 0] = " << dt1.getLocEl(idxs1).value() << std::endl;
  std::cout << my_env.proc_id << ": t1.id = " << dt1.getID() << std::endl;

  auto dt2 = dt1.distribute(1);

  qtnh::tidx_tup idxs2 = { 0, 1 };
  std::cout << my_env.proc_id << ": t2[r, 0, 1] = " << dt2.getLocEl(idxs2).value() << std::endl;
  std::cout << my_env.proc_id << ": t2.id = " << dt2.getID() << std::endl;

  qtnh::Tensor& t1 = dt1;
  qtnh::Tensor& t2 = dt2;

  qtnh::wire w1 = std::pair<qtnh::tidx, qtnh::tidx>(0, 0);
  qtnh::Tensor::contract(&t1, &t1, std::vector<qtnh::wire>(1, w1));
  qtnh::Tensor::contract(&t1, &t2, std::vector<qtnh::wire>(1, w1));
  qtnh::Tensor::contract(&t2, &t1, std::vector<qtnh::wire>(1, w1));
  qtnh::Tensor::contract(&t2, &t2, std::vector<qtnh::wire>(1, w1));


  return 0;
}
