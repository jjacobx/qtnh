#include <catch2/catch_test_macros.hpp>
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
    REQUIRE(tp_dense->totDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(tp_symm->totDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(tp_swap->totDims() == qtnh::tidx_tup { 2, 2, 2, 2 });
    REQUIRE(tp_iden->totDims() == qtnh::tidx_tup { 2, 2 });

    REQUIRE(tp_dense->locDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(tp_symm->locDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(tp_swap->locDims() == qtnh::tidx_tup { 2, 2, 2, 2 });
    REQUIRE(tp_iden->locDims() == qtnh::tidx_tup { 2, 2 });

    REQUIRE(tp_dense->disDims() == qtnh::tidx_tup {});
    REQUIRE(tp_symm->disDims() == qtnh::tidx_tup {});
    REQUIRE(tp_swap->disDims() == qtnh::tidx_tup {});
    REQUIRE(tp_iden->disDims() == qtnh::tidx_tup {});
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
//   auto t_swap_u = std::make_unique<SwapTensor>(ENV, 2, 2);
//   auto t_iden_u = std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 2 });
//   auto t_conv_u = std::make_unique<ConvertTensor>(ENV, qtnh::tidx_tup { 2 });

//   SECTION("dense-dense") {
//     // SDenseTensor x SDenseTensor
//     for (auto& cv : gen::dense_vals) {
//       auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
//       auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);

//       auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

//       qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
//       std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

//       REQUIRE(t_r1_u->getDims() == t_r1_dims);
//       TIndexing ti_r1(t_r1_dims);
//       for (auto idxs : ti_r1) {
//         auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
//         REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
//       }
//     }

//     // SDenseTensor x DDenseTensor
//     for (auto& cv : gen::dense_vals) {
//       auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
//       auto t_sden2_u = std::make_unique<DDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els, 0);

//       auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

//       qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
//       std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

//       REQUIRE(t_r1_u->getDims() == t_r1_dims);
//       TIndexing ti_r1(t_r1_dims);
//       for (auto idxs : ti_r1) {
//         auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
//         REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
//       }
//     }

//     // DDenseTensor x SDenseTensor
//     for (auto& cv : gen::dense_vals) {
//       auto t_sden1_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
//       auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);

//       auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

//       qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
//       std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

//       REQUIRE(t_r1_u->getDims() == t_r1_dims);
//       TIndexing ti_r1(t_r1_dims);
//       for (auto idxs : ti_r1) {
//         auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
//         REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
//       }
//     }

//     // DDenseTensor x DDenseTensor
//     for (auto& cv : gen::dense_vals) {
//       auto t_sden1_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
//       auto t_sden2_u = std::make_unique<DDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els, 0);

//       auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

//       qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
//       std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

//       REQUIRE(t_r1_u->getDims() == t_r1_dims);
//       TIndexing ti_r1(t_r1_dims);
//       for (auto idxs : ti_r1) {
//         auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
//         REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
//       }
//     }

//     // Invalid contraction dimensions
//     std::vector<qtnh::tel> els1 { 1.0, 2.0, 3.0, 4.0 };
//     std::vector<qtnh::tel> els2 { 1.0, 2.0, 3.0 };

//     auto t1_u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els1);
//     auto t2_u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 3 }, els2);
//     REQUIRE_THROWS(Tensor::contract(std::move(t1_u), std::move(t2_u), {{ 0, 0 }}));
//   }

//   SECTION("swap-dense") {
//     // Valid swaps
//     for (auto& cv : gen::swap_vals) {
//       auto t_sden_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
//       auto t_swap_u = std::make_unique<SwapTensor>(ENV, cv.t2_info.dims.at(0), cv.t2_info.dims.at(1));

//       auto t_r1_u = Tensor::contract(std::move(t_sden_u), std::move(t_swap_u), cv.wires);

//       qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
//       std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

//       REQUIRE(t_r1_u->getDims() == t_r1_dims);
//       TIndexing ti_r1(t_r1_dims);
//       for (auto idxs : ti_r1) {
//         auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
//         REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
//       }
//     }

//     // Asymmetric swaps
//     for (auto& cv : gen::invalid_swaps) {
//       auto t_sden_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
//       auto t_swap_u = std::make_unique<SwapTensor>(ENV, cv.t2_info.dims.at(0), cv.t2_info.dims.at(1));

//       REQUIRE_THROWS(Tensor::contract(std::move(t_sden_u), std::move(t_swap_u), cv.wires));
//     }
//   }

//   SECTION("identity-dense") {
//     // Valid identities
//     for (auto& cv : gen::id_vals) {
//       auto t_sden_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
//       auto t_id_u = std::make_unique<IdentityTensor>(ENV, utils::split_dims(cv.t2_info.dims, cv.t2_info.dims.size() / 2).first);

//       auto t_r1_u = Tensor::contract(std::move(t_sden_u), std::move(t_id_u), cv.wires);

//       qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
//       std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

//       REQUIRE(t_r1_u->getDims() == t_r1_dims);
//       TIndexing ti_r1(t_r1_dims);
//       for (auto idxs : ti_r1) {
//         auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
//         REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
//       }
//     }

//     // Invalid contraction dimensions
//     std::vector<qtnh::tel> els1 { 1.0, 2.0, 3.0, 4.0 };

//     auto t1_u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els1);
//     auto t2_u = std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 3 });
//     REQUIRE_THROWS(Tensor::contract(std::move(t1_u), std::move(t2_u), {{ 0, 0 }}));
//   }

//   SECTION("convert-dense") {
//     // The only convert contractions that can be tested serially are tensor products
//     // Also test for invalid convert contractions

//     qtnh::tidx_tup dims { 2, 2 };
//     std::vector<qtnh::tel> els { 1.0i, 2.0i, 3.0i, 4.0i };
//     auto t_sden_u = std::make_unique<SDenseTensor>(ENV, dims, els);
//     auto t_dden_u = std::make_unique<DDenseTensor>(ENV, dims, els, 0);
//     auto t_conv1_u = std::make_unique<ConvertTensor>(ENV, qtnh::tidx_tup {});
//     auto t_conv2_u = std::make_unique<ConvertTensor>(ENV, qtnh::tidx_tup {});

//     // SDenseTensor to DDenseTensor
//     auto t_r1_u = Tensor::contract(std::move(t_sden_u), std::move(t_conv1_u), {});
//     REQUIRE(dynamic_cast<SDenseTensor*>(t_r1_u.get()) == nullptr);
//     REQUIRE(dynamic_cast<DDenseTensor*>(t_r1_u.get()) != nullptr);

//     REQUIRE(t_r1_u->getDims() == dims);
//     TIndexing ti_r1(dims);
//     for (auto idxs : ti_r1) {
//       auto el = els.at(utils::idxs_to_i(idxs, dims));
//       REQUIRE(utils::equal(t_r1_u->getLocEl(idxs).value(), el));
//     }

//     // DDenseTensor to SDenseTensor
//     auto t_r2_u = Tensor::contract(std::move(t_dden_u), std::move(t_conv2_u), {});
//     REQUIRE(dynamic_cast<SDenseTensor*>(t_r2_u.get()) != nullptr);
//     REQUIRE(dynamic_cast<DDenseTensor*>(t_r2_u.get()) == nullptr);

//     REQUIRE(t_r2_u->getDims() == dims);
//     TIndexing ti_r2(dims);
//     for (auto idxs : ti_r2) {
//       auto el = els.at(utils::idxs_to_i(idxs, dims));
//       REQUIRE(utils::equal(t_r2_u->getLocEl(idxs).value(), el));
//     }
//   }
}
