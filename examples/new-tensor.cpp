#include <iostream>

#include "qtnh.hpp"

using namespace std::complex_literals;
using namespace qtnh::ops;

int main() {
  qtnh::QTNHEnv my_env;

  std::vector<qtnh::tel> dt1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  std::vector<qtnh::tel> dt2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

  qtnh::tidx_tup dt1_dims = { 2, 2, 2 };
  qtnh::tidx_tup dt2_dims = { 4, 2 };

  if (my_env.proc_id == 0) std::cout << "Dims: " << dt1_dims << std::endl;

  qtnh::SDenseTensor dt1(my_env, dt1_dims, dt1_els);
  qtnh::SDenseTensor dt2(my_env, dt2_dims, dt2_els);


  qtnh::tidx_tup idxs1 = { 0, 1, 0 };
  std::cout << my_env.proc_id << ": t1[0, 1, 0] = " << dt1.getLocEl(idxs1).value_or(std::nan("1")) << std::endl;
  std::cout << my_env.proc_id << ": t1.id = " << dt1.getID() << std::endl;

  qtnh::tidx_tup idxs2 = { 1, 0 };
  std::cout << my_env.proc_id << ": t2[1, 0] = " << dt2.getLocEl(idxs2).value_or(std::nan("1")) << std::endl;
  std::cout << my_env.proc_id << ": t2.id = " << dt2.getID() << std::endl;

  auto& dt3 = *dt1.distribute(1);

  qtnh::tidx_tup idxs3 = { 0, 1 };
  std::cout << my_env.proc_id << ": t3[r, 0, 1] = " << dt3.getLocEl(idxs3).value_or(std::nan("1")) << std::endl;
  std::cout << my_env.proc_id << ": t3.id = " << dt3.getID() << std::endl;

  qtnh::Tensor& t1 = dt1;
  qtnh::Tensor& t2 = dt2;
  qtnh::Tensor& t3 = dt3;

  qtnh::wire w1 = std::pair<qtnh::tidx, qtnh::tidx>(0, 1);
  auto t4r = qtnh::Tensor::contract(&t1, &t2, std::vector<qtnh::wire>(1, w1));
  auto& t4 = *t4r;

  qtnh::tidx_tup idxs4 = { 0, 0, 0 };
  std::cout << my_env.proc_id << ": t4[0, 0, 0] = " << t4.getLocEl(idxs4).value_or(std::nan("1")) << std::endl;

  qtnh::wire w2 = std::pair<qtnh::tidx, qtnh::tidx>(1, 1);
  auto t5r = qtnh::Tensor::contract(&t3, &t1, std::vector<qtnh::wire>(1, w2));
  auto& t5 = *t5r;

  qtnh::tidx_tup idxs5 = { 0, 0, 0 };
  std::cout << my_env.proc_id << ": t5[r, 0, 0, 0] = " << t5.getLocEl(idxs5).value_or(std::nan("1")) << std::endl;

  qtnh::wire w3 = std::pair<qtnh::tidx, qtnh::tidx>(1, 1);
  auto t6r = qtnh::Tensor::contract(&t1, &t3, std::vector<qtnh::wire>(1, w3));
  auto& t6 = *t6r;

  qtnh::tidx_tup idxs6 = { 0, 0, 0 };
  std::cout << my_env.proc_id << ": t6[r, 0, 0, 0] = " << t6.getLocEl(idxs6).value_or(std::nan("1")) << std::endl;

  auto& dt7 = *dt2.distribute(1);
  qtnh::Tensor& t7 = dt7;

  qtnh::tidx_tup idxs7 = { 0 };
  std::cout << my_env.proc_id << ": t7[r, 0] = " << t7.getLocEl(idxs7).value_or(std::nan("1")) << std::endl;

  qtnh::wire w4 = std::pair<qtnh::tidx, qtnh::tidx>(1, 1);
  auto t8r = qtnh::Tensor::contract(&t3, &t7, std::vector<qtnh::wire>(1, w4));
  auto& t8 = *t8r;

  qtnh::tidx_tup idxs8 = { 0 };
  std::cout << my_env.proc_id << ": t8[r1, r2, 0] = " << t8.getLocEl(idxs8).value_or(std::nan("1")) << std::endl;

  auto dt9 = qtnh::SDenseTensor(my_env, dt1_dims, dt1_els);
  auto& dt10 = *dt9.distribute(1);
  dt10.scatter(1);

  qtnh::tidx_tup idxs10 = { 0 };
  std::cout << my_env.proc_id << ": dt10[r1, r2, 0] = " << dt10.getLocEl(idxs10).value_or(std::nan("1")) << std::endl;

  dt10.gather(2);
  auto& dt11 = dt10;

  qtnh::tidx_tup idxs11 = { 0, 0, 0 };
  std::cout << my_env.proc_id << ": dt11[r1, 0, 0] = " << dt11.getLocEl(idxs11).value_or(std::nan("1")) << std::endl;

  dt11.scatter(2);
  auto& dt12 = *dt11.share();

  qtnh::tidx_tup idxs12 = { 0, 0, 0 };
  std::cout << my_env.proc_id << ": dt12[0, 0, 0] = " << dt12.getLocEl(idxs12).value_or(std::nan("1")) << std::endl;

  qtnh::SDenseTensor dt13(my_env, dt1_dims, dt1_els);
  qtnh::SDenseTensor dt14(my_env, dt2_dims, dt2_els);

  auto t15r = qtnh::Tensor::contract(&dt13, &dt14, std::vector<qtnh::wire>(0));
  auto& t15 = *t15r;
  std::cout << my_env.proc_id << ": dt15[1, 1, 1, 3, 1] = " << t15.getLocEl({1, 1, 1, 3, 1}).value_or(std::nan("1")) << std::endl;

  qtnh::SDenseTensor dt16(my_env, dt1_dims, dt1_els);
  std::cout << my_env.proc_id << ": dt16[0, 0, 1] = " << dt16.getLocEl({0, 0, 1}).value_or(std::nan("1")) << std::endl;

  dt16.swap(1, 2);
  std::cout << my_env.proc_id << ": dt16[0, 0, 1] = " << dt16.getLocEl({0, 0, 1}).value_or(std::nan("1")) << std::endl;

  return 0;
}
