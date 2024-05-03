#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "core/utils.hpp"
#include "tensor/indexing.hpp"
#include "tensor/network.hpp"

#include "gen/random-tn.hpp"

using namespace qtnh;
using namespace std::complex_literals;

QTNHEnv ENV;

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

bool eq(qtnh::tel a, qtnh::tel b, double delta = 1E-5) {
  return (std::abs(a.real() - b.real()) < delta) && (std::abs(a.imag() - b.imag()) < delta);
}

TEST_CASE("contract-tensor-network-test") {
  qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  std::vector<qtnh::tel> t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
  auto t1u = std::make_unique<qtnh::SDenseTensor>(ENV, t1_dims, t1_els);

  qtnh::tidx_tup t2_dims = { 2, 4 };
  std::vector<qtnh::tel> t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  auto t2u = std::make_unique<qtnh::SDenseTensor>(ENV, t2_dims, t2_els);

  qtnh::TensorNetwork tn;
  auto t1_id = tn.insertTensor(std::move(t1u));
  auto t2_id = tn.insertTensor(std::move(t2u));

  std::vector<qtnh::wire> wires1(1, {1, 0});
  tn.createBond(t1_id, t2_id, wires1);

  REQUIRE_NOTHROW(tn.contractAll());
}

TEST_CASE("contract-tensor-network-validation") {
  qtnh::tidx_tup t1_dims = { 2, 2, 2 };
  std::vector<qtnh::tel> t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
  auto t1u = std::make_unique<qtnh::SDenseTensor>(ENV, t1_dims, t1_els);

  qtnh::tidx_tup t2_dims = { 2, 4 };
  std::vector<qtnh::tel> t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
  auto t2u = std::make_unique<qtnh::SDenseTensor>(ENV, t2_dims, t2_els);

  qtnh::tidx_tup tr_dims = { 2, 2, 4 };
  std::vector<qtnh::tel> tr_els = { 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 
                                    2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i };
  qtnh::SDenseTensor tr(ENV, tr_dims, tr_els);

  qtnh::TensorNetwork tn;
  auto t1_id = tn.insertTensor(std::move(t1u));
  auto t2_id = tn.insertTensor(std::move(t2u));

  std::vector<qtnh::wire> wires1(1, {1, 0});
  tn.createBond(t1_id, t2_id, wires1);

  auto res_id = tn.contractAll();

  REQUIRE(equal(tr, tn.getTensor(res_id)));
}

TEST_CASE("tn-contraction") {
  SECTION("2-tensors") {
    for (auto& tnv : gen::tn2_vals) {
      TensorNetwork tn;
      
      // maps between tensor ids in Python and in TensorNetwork
      std::vector<qtnh::uint> maps(tnv.t_infos.size());
      std::size_t i = 0;

      for (auto& t_info : tnv.t_infos) {
        maps.at(i++) = tn.createTensor<SDenseTensor>(ENV, t_info.dims, t_info.els);
      }

      for (auto b_info : tnv.b_infos) {
        tn.createBond(maps.at(b_info.t1_idx), maps.at(b_info.t2_idx), b_info.wires);
      }

      auto res_id = tn.contractAll();
      auto t_res_u = tn.extractTensor(res_id);

      qtnh::tidx_tup t_res_dims = tnv.result_info.dims;
      std::vector<qtnh::tel> t_res_els = tnv.result_info.els;

      REQUIRE(t_res_u->getDims() == t_res_dims);
      TIndexing ti_res(t_res_dims);
      for (auto idxs : ti_res) {
        auto el = t_res_els.at(utils::idxs_to_i(idxs, t_res_dims));
        REQUIRE(eq(t_res_u->getLocEl(idxs).value(), el));
      }
    }
  }

  SECTION("3-tensors") {
    for (auto& tnv : gen::tn3_vals) {
      TensorNetwork tn;
      
      // maps between tensor ids in Python and in TensorNetwork
      std::vector<qtnh::uint> maps(tnv.t_infos.size());
      std::size_t i = 0;

      for (auto& t_info : tnv.t_infos) {
        maps.at(i++) = tn.createTensor<SDenseTensor>(ENV, t_info.dims, t_info.els);
      }

      for (auto b_info : tnv.b_infos) {
        tn.createBond(maps.at(b_info.t1_idx), maps.at(b_info.t2_idx), b_info.wires);
      }

      auto res_id = tn.contractAll();
      auto t_res_u = tn.extractTensor(res_id);

      qtnh::tidx_tup t_res_dims = tnv.result_info.dims;
      std::vector<qtnh::tel> t_res_els = tnv.result_info.els;

      REQUIRE(t_res_u->getDims() == t_res_dims);
      TIndexing ti_res(t_res_dims);
      for (auto idxs : ti_res) {
        auto el = t_res_els.at(utils::idxs_to_i(idxs, t_res_dims));

        // ! Weird behaviour for larger contractions - the accuracy gets low quickly. 
        // if (!eq(t_res_u->getLocEl(idxs).value(), el, 1E-2)) {
        //   using namespace qtnh::ops;
        //   std::cout << "ERROR: P" << ENV.proc_id << ", idxs = " << idxs << "\n";
        //   std::cout << "El: " << t_res_u->getLocEl(idxs).value() << " != " << el << "\n";
        //   std::cout << "Real diff: " << std::abs(t_res_u->getLocEl(idxs).value().real() - el.real()) << "\n";
        //   std::cout << "Imag diff: " << std::abs(t_res_u->getLocEl(idxs).value().imag() - el.imag()) << "\n"; 
        // }


        REQUIRE(eq(t_res_u->getLocEl(idxs).value(), el, 1E-2));
      }
    }
  }
}
