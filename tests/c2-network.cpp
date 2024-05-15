#include <catch2/catch_test_macros.hpp>
#include <iostream>

#include "core/utils.hpp"
#include "tensor/indexing.hpp"
#include "tensor/network.hpp"
#include "tensor/special.hpp"

#include "gen/random-tn.hpp"
#include "gen/qft.hpp"

using namespace qtnh;
using namespace std::complex_literals;

QTNHEnv ENV;

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
        REQUIRE(utils::equal(t_res_u->getLocEl(idxs).value(), el));
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


        REQUIRE(utils::equal(t_res_u->getLocEl(idxs).value(), el, 1E-2));
      }
    }
  }

  SECTION("4-tensors") {
    for (auto& tnv : gen::tn4_vals) {
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
        REQUIRE(utils::equal(t_res_u->getLocEl(idxs).value(), el, 1E-2));
      }
    }
  }

  SECTION("4-circular-tensors") {
    for (auto& tnv : gen::tn4c_vals) {
      TensorNetwork tn;

      // maps between tensor ids in Python and in TensorNetwork
      std::vector<qtnh::uint> maps(tnv.t_infos.size());
      std::size_t i = 0;

      for (auto& t_info : tnv.t_infos) {
        maps.at(i++) = tn.createTensor<SDenseTensor>(ENV, t_info.dims, t_info.els);
      }

      std::vector<qtnh::uint> b_ord;

      for (auto b_info : tnv.b_infos) {
        auto bid = tn.createBond(maps.at(b_info.t1_idx), maps.at(b_info.t2_idx), b_info.wires);
        b_ord.push_back(bid);
      }

      auto res_id = tn.contractAll(b_ord);
      auto t_res_u = tn.extractTensor(res_id);

      qtnh::tidx_tup t_res_dims = tnv.result_info.dims;
      std::vector<qtnh::tel> t_res_els = tnv.result_info.els;

      REQUIRE(t_res_u->getDims() == t_res_dims);
      TIndexing ti_res(t_res_dims);
      for (auto idxs : ti_res) {
        auto el = t_res_els.at(utils::idxs_to_i(idxs, t_res_dims));
        REQUIRE(utils::equal(t_res_u->getLocEl(idxs).value(), el, 1E-2));
      }
    }
  }
}

TEST_CASE("qft") {
  SECTION("2-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 2, 0);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());
    REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-2));
  }

  SECTION("3-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 3, 0);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());
    REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-2));
  }

  SECTION("4-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 4, 0);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());
    REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-2));
  }

  SECTION("5-qubits") {
    TensorNetwork tn;
    auto con_ord = gen::qft(ENV, tn, 5, 0);

    auto id = tn.contractAll(con_ord);
    auto tfu = tn.extractTensor(id);

    auto idxs = utils::i_to_idxs(0, tfu->getLocDims());
    REQUIRE(utils::equal(tfu->getLocEl(idxs).value(), 1, 1E-2));
  }
}