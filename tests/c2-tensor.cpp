#include <catch2/catch_test_macros.hpp>
#include <iostream>
#include <memory>
#include <vector>

#include "core/utils.hpp"
#include "tensor/tensor.hpp"
#include "tensor/dense.hpp"
#include "tensor/symm.hpp"
#include "tensor/diag.hpp"

#include "gen/random-tensors.hpp"

using namespace qtnh;
using namespace std::complex_literals;

QTNHEnv ENV;

TEST_CASE("tensor-construction") {
  SECTION("dense-tensor") {
    REQUIRE_NOTHROW(DenseTensor::make(ENV, {}, { 2, 2 }, { 1.0i, 2.0i, 3.0i, 4.0i }));
    REQUIRE_NOTHROW(DenseTensor::make(ENV, {}, { 2, 2 }, { 1.0i, 2.0i, 3.0i, 4.0i }, { 1, 1, 0 }));
  }

  SECTION("symmetric-tensor") {
    REQUIRE_NOTHROW(SymmTensor::make(ENV, {}, { 2, 2 }, 0, { 1.0i, 2.0i, 3.0i, 4.0i }));
    REQUIRE_NOTHROW(SymmTensor::make(ENV, {}, { 2, 2 }, 0, { 1.0i, 2.0i, 3.0i, 4.0i }, { 1, 1, 0 }));
  }

  SECTION("swap-tensor") {
    REQUIRE_NOTHROW(SwapTensor::make(ENV, 2, 0));
    REQUIRE_NOTHROW(SwapTensor::make(ENV, 2, 0, { 1, 1, 0 }));
  }

  // TODO: Implement diagonal tensors. 
  // SECTION("diagonal-tensor") {
  //   REQUIRE_NOTHROW(DiagTensor::make(ENV, {}, { 2, 2 }, 0, { 1.0i, 2.0i, 3.0i, 4.0i }, 0));
  //   REQUIRE_NOTHROW(DiagTensor::make(ENV, {}, { 2, 2 }, 0, { 1.0i, 2.0i, 3.0i, 4.0i }, 0, { 1, 1, 0 }));
  // }

  SECTION("identity-tensor") {
    REQUIRE_NOTHROW(IdenTensor::make(ENV, {}, { 2, 2 }, 0, 0));
    REQUIRE_NOTHROW(IdenTensor::make(ENV, {}, { 2, 2 }, 0, 0, { 1, 1, 0 }));
  }
}

TEST_CASE("tensor-accessors") {
  auto tp_dense = DenseTensor::make(ENV, {}, { 2, 2 }, { 1.0i, 2.0i, 3.0i, 4.0i });
  auto tp_symm = SymmTensor::make(ENV, {}, { 2, 2 }, 0, { 1.0i, 2.0i, 3.0i, 4.0i });
  auto tp_swap = SwapTensor::make(ENV, 2, 0);
  auto tp_iden = IdenTensor::make(ENV, {}, { 2, 2 }, 0, 0);

  SECTION("get-dims") {
    REQUIRE(tp_dense->totDims() == tidx_tup { 2, 2 });
    REQUIRE(tp_symm->totDims() == tidx_tup { 2, 2 });
    REQUIRE(tp_swap->totDims() == tidx_tup { 2, 2, 2, 2 });
    REQUIRE(tp_iden->totDims() == tidx_tup { 2, 2 });

    REQUIRE(tp_dense->locDims() == tidx_tup { 2, 2 });
    REQUIRE(tp_symm->locDims() == tidx_tup { 2, 2 });
    REQUIRE(tp_swap->locDims() == tidx_tup { 2, 2, 2, 2 });
    REQUIRE(tp_iden->locDims() == tidx_tup { 2, 2 });

    REQUIRE(tp_dense->disDims() == tidx_tup {});
    REQUIRE(tp_symm->disDims() == tidx_tup {});
    REQUIRE(tp_swap->disDims() == tidx_tup {});
    REQUIRE(tp_iden->disDims() == tidx_tup {});
  }

  SECTION("get-size") {
    REQUIRE(tp_dense->totSize() == 4);
    REQUIRE(tp_symm->totSize() == 4);
    REQUIRE(tp_swap->totSize() == 16);
    REQUIRE(tp_iden->totSize() == 4);

    REQUIRE(tp_dense->locSize() == 4);
    REQUIRE(tp_symm->locSize() == 4);
    REQUIRE(tp_swap->locSize() == 16);
    REQUIRE(tp_iden->locSize() == 4);

    REQUIRE(tp_dense->disSize() == 1);
    REQUIRE(tp_symm->disSize() == 1);
    REQUIRE(tp_swap->disSize() == 1);
    REQUIRE(tp_iden->disSize() == 1);
  }

  std::vector<qtnh::tel> els_dense { 
    1.0i, 2.0i, 
    3.0i, 4.0i 
  };
  std::vector<qtnh::tel> els_symm { 
    1.0i, 2.0i, 
    3.0i, 4.0i 
  };
  std::vector<qtnh::tel> els_swap { 
    1.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 
    0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 1.0 
  };
  std::vector<qtnh::tel> els_iden { 
    1.0, 0.0, 
    0.0, 1.0 
  };

  SECTION("get-element") {
    TIndexing ti_dense(tp_dense->totDims());
    for (auto idxs : ti_dense.tup()) {
      auto el = els_dense.at(utils::idxs_to_i(idxs, tp_dense->totDims()));
      REQUIRE(tp_dense->at(idxs) == el);
    }

    TIndexing ti_symm(tp_symm->totDims());
    for (auto idxs : ti_symm.tup()) {
      auto el = els_symm.at(utils::idxs_to_i(idxs, tp_symm->totDims()));
      REQUIRE(tp_symm->at(idxs) == el);
    }

    TIndexing ti_swap(tp_swap->totDims());
    for (auto idxs : ti_swap.tup()) {
      auto el = els_swap.at(utils::idxs_to_i(idxs, tp_swap->totDims()));
      REQUIRE(tp_swap->at(idxs) == el);
    }

    TIndexing ti_iden(tp_iden->totDims());
    for (auto idxs : ti_iden.tup()) {
      auto el = els_iden.at(utils::idxs_to_i(idxs, tp_iden->totDims()));
      REQUIRE(tp_iden->at(idxs) == el);
    }
  }

  SECTION("set-element") {
    REQUIRE_NOTHROW(tp_dense->at({ 0, 0 }) = 0.1i);
    REQUIRE(tp_dense->at({ 0, 0 }) == 0.1i);

    REQUIRE_NOTHROW(tp_symm->at({ 0, 0 }) = 0.1i);
    REQUIRE(tp_symm->at({ 0, 0 }) == 0.1i);
  }
}

TEST_CASE("tensor-contraction") {
  SECTION("dense-dense") {
    for (auto& cv : gen::dense_vals) {
      tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
      tptr tp2 = DenseTensor::make(ENV, {}, cv.t2_info.dims, std::vector<tel>(cv.t2_info.els));
      tptr tp3;

      REQUIRE_NOTHROW(tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires));

      auto dims = cv.t3_info.dims;
      auto els = cv.t3_info.els;

      REQUIRE(tp3->totDims() == dims);
      TIndexing ti(dims);
      for (auto idxs : ti.tup()) {
        auto el = els.at(utils::idxs_to_i(idxs, dims));
        REQUIRE(utils::equal(tp3->at(idxs), el));
      }
    }

    // Invalid contraction dimensions
    tptr tp1 = DenseTensor::make(ENV, {}, { 2, 2 }, { 1.0, 2.0, 3.0, 4.0 });
    tptr tp2 = DenseTensor::make(ENV, {}, { 3 }, { 1.0, 2.0, 3.0 });
    REQUIRE_THROWS(Tensor::contract(std::move(tp1), std::move(tp2), {{ 0, 0 }}));
  }

  SECTION("dense-swap") {
    for (auto& cv : gen::swap_vals) {
      tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
      tptr tp2 = SwapTensor::make(ENV, cv.t2_info.dims.at(0), 0);
      tptr tp3;

      REQUIRE_NOTHROW(tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires));

      auto dims = cv.t3_info.dims;
      auto els = cv.t3_info.els;

      REQUIRE(tp3->totDims() == dims);
      TIndexing ti(dims);
      for (auto idxs : ti.tup()) {
        auto el = els.at(utils::idxs_to_i(idxs, dims));

        // TODO: Implement symmetric tensor contraction. 
        // REQUIRE(utils::equal(tp3->at(idxs), el));
      }
    }

    // TODO: Think what to do with asymmetric swaps. 
    for (auto& cv : gen::invalid_swaps) {
      tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
      tptr tp2 = SwapTensor::make(ENV, cv.t2_info.dims.at(0), 0);
      tptr tp3;

      REQUIRE_THROWS(tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires));
    }
  }

  SECTION("dense-identity") {
    for (auto& cv : gen::id_vals) {
      tptr tp1 = DenseTensor::make(ENV, {}, cv.t1_info.dims, std::vector<tel>(cv.t1_info.els));
      tptr tp2 = IdenTensor::make(ENV, {}, cv.t2_info.dims, 0, 0);
      tptr tp3;

      REQUIRE_NOTHROW(tp3 = Tensor::contract(std::move(tp1), std::move(tp2), cv.wires));

      auto dims = cv.t3_info.dims;
      auto els = cv.t3_info.els;

      // REQUIRE(tp3->totDims() == dims);
      TIndexing ti(dims);
      for (auto idxs : ti.tup()) {
        auto el = els.at(utils::idxs_to_i(idxs, dims));

        // TODO: Implement symmetric tensor contraction. 
        // REQUIRE(utils::equal(tp3->at(idxs), el));
      }
    }
  }
}
