#include <catch2/catch_test_macros.hpp>
#include "tensor/indexing.hpp"
#include "tensor/network.hpp"

using namespace std::complex_literals;

bool equal(const qtnh::Tensor& t1, const qtnh::Tensor& t2) {
  if (t1.getLocDims() != t2.getLocDims()) { 
    return false; 
  }

  qtnh::TIndexing ti(t1.getLocDims());
  for (auto idxs : ti) {
    if (t1.getLocEl(idxs) != t2.getLocEl(idxs)) { 
      return false; 
    }
  }

  return true;
}

TEST_CASE("create-tensor-validation") {
  qtnh::QTNHEnv env;
  qtnh::tidx_tup t1_dims = { 2, 2 };
  std::vector<qtnh::tel> t11_els (4, 0.0);
  std::vector<qtnh::tel> t12_els = { 0.0, 0.0, 0.0, 0.0 };
  qtnh::SDenseTensor t11(env, t1_dims, t11_els);
  qtnh::SDenseTensor t12(env, t1_dims, t12_els);

  qtnh::tidx_tup t2_dims = { 1 };
  std::vector<qtnh::tel> t21_els(1, 1.0);
  std::vector<qtnh::tel> t22_els = { 1.0 };

  qtnh::SDenseTensor t21(env, t2_dims, t21_els);
  qtnh::SDenseTensor t22(env, t2_dims, t22_els);

  REQUIRE(equal(t11, t12));
  REQUIRE(equal(t21, t22));
  REQUIRE(!equal(t11, t21));
  REQUIRE(!equal(t12, t22));

  REQUIRE(t11.getID() == 1);
  REQUIRE(t12.getID() == 2);
  REQUIRE(t21.getID() == 3);
  REQUIRE(t22.getID() == 4);
}

TEST_CASE("access-tensor-validation") {
  qtnh::QTNHEnv env;

  qtnh::tidx_tup t1_dims = { 2, 2 };
  std::vector<qtnh::tel> t1_els = { 0.0, 1.0, 2.0, 3.0 };
  qtnh::SDenseTensor t1(env, t1_dims, t1_els);

  std::vector<qtnh::tel> t2_els = { 0.0, 2.0, 1.0, 3.0 };
  qtnh::tidx_tup t2_dims = { 2, 2 };
  qtnh::SDenseTensor t2(env, t2_dims, t2_els);

  auto el = t1[{1, 0}];
  t1[{1, 0}] = t1[{0, 1}];
  t1[{0, 1}] = el;

  REQUIRE(equal(t1, t2));
}


TEST_CASE("contract-tensor-network-validation") {
  qtnh::QTNHEnv env;

  qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  std::vector<qtnh::tel> t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
  qtnh::SDenseTensor t1(env, t1_dims, t1_els);

  qtnh::tidx_tup t2_dims = { 2, 4 };
  std::vector<qtnh::tel> t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  qtnh::SDenseTensor t2(env, t2_dims, t2_els);

  qtnh::tidx_tup tr_dims = { 2, 2, 4 };
  std::vector<qtnh::tel> tr_els = { 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 
                                    2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i };
  qtnh::SDenseTensor tr(env, tr_dims, tr_els);

  qtnh::TensorNetwork tn;
  auto t1_id = tn.insertTensor(&t1);
  auto t2_id = tn.insertTensor(&t2);

  std::vector<qtnh::wire> wires1(1, {1, 0});
  tn.createBond(t1_id, t2_id, wires1);

  auto res_id = tn.contractAll();

  REQUIRE(equal(tr, tn.getTensor(res_id)));
}
