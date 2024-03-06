#include <catch2/catch_test_macros.hpp>
#include "tensor/network.hpp"

using namespace std::complex_literals;

TEST_CASE("create-tensor-test") {
  qtnh::QTNHEnv env;

  std::vector<qtnh::tel> x_els = { 0.0, 1.0, 1.0, 0.0 };
  std::vector<qtnh::tel> y_els = { 0.0, -1.0i, 1.0i, 0.0 };
  std::vector<qtnh::tel> copy_els = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

  qtnh::tidx_tup x_dims = { 2, 2 };
  qtnh::tidx_tup y_dims = { 2, 2 };
  qtnh::tidx_tup copy_dims = { 2, 2, 2 };

  qtnh::SDenseTensor* x;
  qtnh::SDenseTensor* y;
  qtnh::SDenseTensor* copy;

  REQUIRE_NOTHROW(x = new qtnh::SDenseTensor(env, x_dims, x_els));
  REQUIRE_NOTHROW(y = new qtnh::SDenseTensor(env, y_dims, y_els));
  REQUIRE_NOTHROW(copy = new qtnh::SDenseTensor(env, copy_dims, copy_els));
}

TEST_CASE("access-tensor-test") {
  qtnh::QTNHEnv env;
  std::vector<qtnh::tel> t_els = { 0.0, 1.0, 2.0, 3.0 };
  qtnh::tidx_tup t_dims = { 2, 2 };
  qtnh::SDenseTensor t(env, t_dims, t_els);

  qtnh::tidx_tup i1 = { 0, 1 };
  qtnh::tidx_tup i2 = { 1, 0 };
  qtnh::tel c;

  REQUIRE_NOTHROW(c = t[i1]);
  REQUIRE_NOTHROW(t[i2] = c);
}

TEST_CASE("contract-tensor-network-test") {
  qtnh::QTNHEnv env;

  qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  std::vector<qtnh::tel> t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
  qtnh::SDenseTensor t1(env, t1_dims, t1_els);

  qtnh::tidx_tup t2_dims = { 2, 4 };
  std::vector<qtnh::tel> t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  qtnh::SDenseTensor t2(env, t2_dims, t2_els);

  qtnh::TensorNetwork tn;
  auto t1_id = tn.insertTensor(&t1);
  auto t2_id = tn.insertTensor(&t2);

  std::vector<qtnh::wire> wires1(1, {1, 0});
  tn.createBond(t1_id, t2_id, wires1);

  REQUIRE_NOTHROW(tn.contractAll());
}
