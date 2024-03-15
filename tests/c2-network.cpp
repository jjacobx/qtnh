#include <catch2/catch_test_macros.hpp>


#include "tensor/indexing.hpp"
#include "tensor/network.hpp"

using namespace qtnh;
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

TEST_CASE("contract-tensor-network-test") {
  qtnh::QTNHEnv env;

  qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  std::vector<qtnh::tel> t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
  auto t1u = std::make_unique<qtnh::SDenseTensor>(env, t1_dims, t1_els);

  qtnh::tidx_tup t2_dims = { 2, 4 };
  std::vector<qtnh::tel> t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  auto t2u = std::make_unique<qtnh::SDenseTensor>(env, t2_dims, t2_els);

  qtnh::TensorNetwork tn;
  auto t1_id = tn.insertTensor(std::move(t1u));
  auto t2_id = tn.insertTensor(std::move(t2u));

  std::vector<qtnh::wire> wires1(1, {1, 0});
  tn.createBond(t1_id, t2_id, wires1);

  REQUIRE_NOTHROW(tn.contractAll());
}

TEST_CASE("contract-tensor-network-validation") {
  qtnh::QTNHEnv env;

  qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  std::vector<qtnh::tel> t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
  auto t1u = std::make_unique<qtnh::SDenseTensor>(env, t1_dims, t1_els);

  qtnh::tidx_tup t2_dims = { 2, 4 };
  std::vector<qtnh::tel> t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  auto t2u = std::make_unique<qtnh::SDenseTensor>(env, t2_dims, t2_els);

  qtnh::tidx_tup tr_dims = { 2, 2, 4 };
  std::vector<qtnh::tel> tr_els = { 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 
                                    2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i };
  qtnh::SDenseTensor tr(env, tr_dims, tr_els);

  qtnh::TensorNetwork tn;
  auto t1_id = tn.insertTensor(std::move(t1u));
  auto t2_id = tn.insertTensor(std::move(t2u));

  std::vector<qtnh::wire> wires1(1, {1, 0});
  tn.createBond(t1_id, t2_id, wires1);

  auto res_id = tn.contractAll();

  REQUIRE(equal(tr, tn.getTensor(res_id)));
}

