#include <iostream>

#include "qtnh.hpp"

using namespace qtnh;
using namespace qtnh::ops;

using namespace std::complex_literals;

int main() {
  QTNHEnv env;

  qtnh::tidx_tup dt1_dims = { 2, 2, 2 };
  qtnh::tidx_tup dt2_dims = { 4, 2 };

  std::vector<qtnh::tel> dt1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
  std::vector<qtnh::tel> dt2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

  auto t1u = std::make_unique<SDenseTensor>(env, dt1_dims, dt1_els);
  auto t2u = std::make_unique<SDenseTensor>(env, dt2_dims, dt2_els);

  std::cout << env.proc_id << " | T1[0, 1, 0] = " << t1u->getLocEl({ 0, 1, 0 }).value_or(std::nan("1")) << "\n";
  std::cout << env.proc_id << " | T2[1, 0] = " << t2u->getLocEl({ 1, 0 }).value_or(std::nan("1")) << "\n";

  auto t3u = std::unique_ptr<DDenseTensor>(t1u->distribute(1));
  std::cout << env.proc_id << " | T3[r, 0, 1] = " << t3u->getLocEl({ 0, 1 }).value_or(std::nan("1")) << "\n";

  auto t4u = Tensor::contract(std::move(t1u), std::move(t2u), {{ 0, 1 }});
  std::cout << env.proc_id << " | T4[0, 0, 0] = " << t4u->getLocEl({ 0, 0, 0 }).value_or(std::nan("1")) << "\n";

  auto t5u = std::make_unique<SDenseTensor>(env, dt1_dims, dt1_els);
  auto t6u = std::unique_ptr<DDenseTensor>(t5u->distribute(1));
  
  t6u->scatter(1);
  std::cout << env.proc_id << " | T6[r1, r2, 0] = " << t6u->getLocEl({ 0 }).value_or(std::nan("1")) << "\n";

  t6u->gather(2);
  auto t7u = std::move(t6u);
  std::cout << env.proc_id << " | T7[r1, 0, 0] = " << t7u->getLocEl({ 0, 0, 0 }).value_or(std::nan("1")) << "\n";

  t7u->scatter(2);
  auto t8u = std::unique_ptr<SDenseTensor>(t7u->share());
  std::cout << env.proc_id << " | T8[0, 0, 0] = " << t8u->getLocEl({ 0, 0, 0 }).value_or(std::nan("1")) << "\n";

  auto t9u = std::make_unique<SDenseTensor>(env, dt1_dims, dt1_els);
  auto t10u = std::make_unique<SDenseTensor>(env, dt2_dims, dt2_els);

  auto t11u = Tensor::contract(std::move(t9u), std::move(t10u), {});
  std::cout << env.proc_id << ": T11[1, 1, 1, 3, 1] = " << t11u->getLocEl({1, 1, 1, 3, 1}).value_or(std::nan("1")) << "\n";

  return 0;
}
