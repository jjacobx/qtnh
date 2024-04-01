#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "core/utils.hpp"
#include "tensor/dense.hpp"
#include "tensor/indexing.hpp"
#include "tensor/special.hpp"

#include "gen/random-tensors.hpp"

using namespace qtnh;
using namespace std::complex_literals;

QTNHEnv ENV;

bool eq(qtnh::tel a, qtnh::tel b) {
  return (std::abs(a.real() - b.real()) < 1E-5) && (std::abs(a.imag() - b.imag()) < 1E-5);
}

TEST_CASE("tensor-construction") {
  SECTION("swap-tensor") {
    REQUIRE_NOTHROW(std::make_unique<SwapTensor>(ENV, 2, 2));
  }

  SECTION("identity-tensor") {
    REQUIRE_NOTHROW(std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 2, 2 }));
  }

  SECTION("convert-tensor") {
    REQUIRE_NOTHROW(std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 2, 2 }));
  }

  std::vector<qtnh::tel> els { 1.0i, 2.0i, 3.0i, 4.0i };

  SECTION("shared-dense-tensor") {
    REQUIRE_NOTHROW(std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els));
    REQUIRE_NOTHROW(std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els, false));
  }

  SECTION("distributed-dense-tensor") {
    REQUIRE_NOTHROW(std::make_unique<DDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els, 0));
  }
}

TEST_CASE("tensor-accessors") {
  auto t_swap_u = std::make_unique<SwapTensor>(ENV, 2, 2);
  auto t_iden_u = std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 2 });
  auto t_conv_u = std::make_unique<ConvertTensor>(ENV, qtnh::tidx_tup { 2 });

  std::vector<qtnh::tel> els { 1.0i, 2.0i, 3.0i, 4.0i };
  auto t_sden_u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els);
  auto t_dden_u = std::make_unique<DDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els, 0);

  SECTION("get-dims") {
    REQUIRE(t_swap_u->getDims() == qtnh::tidx_tup { 2, 2, 2, 2 });
    REQUIRE(t_iden_u->getDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(t_conv_u->getDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(t_sden_u->getDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(t_dden_u->getDims() == qtnh::tidx_tup { 2, 2 });

    REQUIRE(t_swap_u->getLocDims() == qtnh::tidx_tup { 2, 2, 2, 2 });
    REQUIRE(t_iden_u->getLocDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(t_conv_u->getLocDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(t_sden_u->getLocDims() == qtnh::tidx_tup { 2, 2 });
    REQUIRE(t_dden_u->getLocDims() == qtnh::tidx_tup { 2, 2 });

    REQUIRE(t_swap_u->getDistDims() == qtnh::tidx_tup {});
    REQUIRE(t_iden_u->getDistDims() == qtnh::tidx_tup {});
    REQUIRE(t_conv_u->getDistDims() == qtnh::tidx_tup {});
    REQUIRE(t_sden_u->getDistDims() == qtnh::tidx_tup {});
    REQUIRE(t_dden_u->getDistDims() == qtnh::tidx_tup {});
  }

  SECTION("get-size") {
    REQUIRE(t_swap_u->getSize() == 16);
    REQUIRE(t_iden_u->getSize() == 4);
    REQUIRE(t_conv_u->getSize() == 4);
    REQUIRE(t_sden_u->getSize() == 4);
    REQUIRE(t_dden_u->getSize() == 4);

    REQUIRE(t_swap_u->getLocSize() == 16);
    REQUIRE(t_iden_u->getLocSize() == 4);
    REQUIRE(t_conv_u->getLocSize() == 4);
    REQUIRE(t_sden_u->getLocSize() == 4);
    REQUIRE(t_dden_u->getLocSize() == 4);

    REQUIRE(t_swap_u->getDistSize() == 1);
    REQUIRE(t_iden_u->getDistSize() == 1);
    REQUIRE(t_conv_u->getDistSize() == 1);
    REQUIRE(t_sden_u->getDistSize() == 1);
    REQUIRE(t_dden_u->getDistSize() == 1);
  }

  std::vector<qtnh::tel> t_swap_els { 
    1.0, 0.0, 0.0, 0.0, 
    0.0, 0.0, 1.0, 0.0, 
    0.0, 1.0, 0.0, 0.0, 
    0.0, 0.0, 0.0, 1.0 
  };
  std::vector<qtnh::tel> t_iden_els { 
    1.0, 0.0, 
    0.0, 1.0 
  };
  std::vector<qtnh::tel> t_conv_els { 
    1.0, 0.0, 
    0.0, 1.0 
  };
  std::vector<qtnh::tel> t_sden_els { 
    1.0i, 2.0i, 
    3.0i, 4.0i 
  };
  std::vector<qtnh::tel> t_dden_els { 
    1.0i, 2.0i, 
    3.0i, 4.0i 
  };

  SECTION("get-element") {
    TIndexing ti_swap(t_swap_u->getDims());
    for (auto idxs : ti_swap) {
      auto el = t_swap_els.at(utils::idxs_to_i(idxs, t_swap_u->getDims()));
      REQUIRE(t_swap_u->getEl(idxs).value() == el);
      REQUIRE(t_swap_u->getLocEl(idxs).value() == el);
      REQUIRE((*t_swap_u)[idxs] == el);
    }

    TIndexing ti_iden(t_iden_u->getDims());
    for (auto idxs : ti_iden) {
      auto el = t_iden_els.at(utils::idxs_to_i(idxs, t_iden_u->getDims()));
      REQUIRE(t_iden_u->getEl(idxs).value() == el);
      REQUIRE(t_iden_u->getLocEl(idxs).value() == el);
      REQUIRE((*t_iden_u)[idxs] == el);
    }

    TIndexing ti_conv(t_conv_u->getDims());
    for (auto idxs : ti_conv) {
      auto el = t_conv_els.at(utils::idxs_to_i(idxs, t_conv_u->getDims()));
      REQUIRE(t_conv_u->getEl(idxs).value() == el);
      REQUIRE(t_conv_u->getLocEl(idxs).value() == el);
      REQUIRE((*t_conv_u)[idxs] == el);
    }

    TIndexing ti_sden(t_sden_u->getDims());
    for (auto idxs : ti_sden) {
      auto el = t_sden_els.at(utils::idxs_to_i(idxs, t_sden_u->getDims()));
      REQUIRE(t_sden_u->getEl(idxs).value() == el);
      REQUIRE(t_sden_u->getLocEl(idxs).value() == el);
      REQUIRE((*t_sden_u)[idxs] == el);
    }

    TIndexing ti_dden(t_dden_u->getDims());
    for (auto idxs : ti_dden) {
      auto el = t_dden_els.at(utils::idxs_to_i(idxs, t_dden_u->getDims()));
      REQUIRE(t_dden_u->getEl(idxs).value() == el);
      REQUIRE(t_dden_u->getLocEl(idxs).value() == el);
      REQUIRE((*t_dden_u)[idxs] == el);
    }
  }

  SECTION("set-element") {
    REQUIRE_NOTHROW(t_sden_u->setEl({ 0, 0 }, 0.1i));
    REQUIRE_NOTHROW(t_sden_u->setLocEl({ 0, 1 }, 0.2i));
    REQUIRE_NOTHROW((*t_sden_u)[{ 1, 0 }] = 0.3i);

    REQUIRE(t_sden_u->getEl({ 0, 0 }) == 0.1i);
    REQUIRE(t_sden_u->getLocEl({ 0, 1 }) == 0.2i);
    REQUIRE((*t_sden_u)[{ 1, 0 }] == 0.3i);

    REQUIRE_NOTHROW(t_dden_u->setEl({ 0, 0 }, 0.1i));
    REQUIRE_NOTHROW(t_dden_u->setLocEl({ 0, 1 }, 0.2i));
    REQUIRE_NOTHROW((*t_dden_u)[{ 1, 0 }] = 0.3i);

    REQUIRE(t_dden_u->getEl({ 0, 0 }) == 0.1i);
    REQUIRE(t_dden_u->getLocEl({ 0, 1 }) == 0.2i);
    REQUIRE((*t_dden_u)[{ 1, 0 }] == 0.3i);
  }
}

TEST_CASE("tensor-contraction") {
  auto t_swap_u = std::make_unique<SwapTensor>(ENV, 2, 2);
  auto t_iden_u = std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 2 });
  auto t_conv_u = std::make_unique<ConvertTensor>(ENV, qtnh::tidx_tup { 2 });

  SECTION("dense-dense") {
    // SDenseTensor x SDenseTensor
    for (auto& cv : gen::dense_vals) {
      auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
      auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);

      auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

      qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
      std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

      REQUIRE(t_r1_u->getDims() == t_r1_dims);
      TIndexing ti_r1(t_r1_dims);
      for (auto idxs : ti_r1) {
        auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
        REQUIRE(eq(t_r1_u->getLocEl(idxs).value(), el));
      }
    }

    // SDenseTensor x DDenseTensor
    for (auto& cv : gen::dense_vals) {
      auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els);
      auto t_sden2_u = std::make_unique<DDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els, 0);

      auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

      qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
      std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

      REQUIRE(t_r1_u->getDims() == t_r1_dims);
      TIndexing ti_r1(t_r1_dims);
      for (auto idxs : ti_r1) {
        auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
        REQUIRE(eq(t_r1_u->getLocEl(idxs).value(), el));
      }
    }

    // DDenseTensor x SDenseTensor
    for (auto& cv : gen::dense_vals) {
      auto t_sden1_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
      auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els);

      auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

      qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
      std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

      REQUIRE(t_r1_u->getDims() == t_r1_dims);
      TIndexing ti_r1(t_r1_dims);
      for (auto idxs : ti_r1) {
        auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
        REQUIRE(eq(t_r1_u->getLocEl(idxs).value(), el));
      }
    }

    // DDenseTensor x DDenseTensor
    for (auto& cv : gen::dense_vals) {
      auto t_sden1_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
      auto t_sden2_u = std::make_unique<DDenseTensor>(ENV, cv.t2_info.dims, cv.t2_info.els, 0);

      auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), cv.wires);

      qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
      std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

      REQUIRE(t_r1_u->getDims() == t_r1_dims);
      TIndexing ti_r1(t_r1_dims);
      for (auto idxs : ti_r1) {
        auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
        REQUIRE(eq(t_r1_u->getLocEl(idxs).value(), el));
      }
    }

    // TODO: test invalid contractions
  }

  SECTION("swap-dense") {
    // Valid swaps
    for (auto& cv : gen::swap_vals) {
      auto t_sden_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
      auto t_swap_u = std::make_unique<SwapTensor>(ENV, cv.t2_info.dims.at(0), cv.t2_info.dims.at(1));

      auto t_r1_u = Tensor::contract(std::move(t_sden_u), std::move(t_swap_u), cv.wires);

      qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
      std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

      REQUIRE(t_r1_u->getDims() == t_r1_dims);
      TIndexing ti_r1(t_r1_dims);
      for (auto idxs : ti_r1) {
        auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
        REQUIRE(eq(t_r1_u->getLocEl(idxs).value(), el));
      }
    }

    // Invalid swaps
    for (auto& cv : gen::invalid_swaps) {
      auto t_sden_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
      auto t_swap_u = std::make_unique<SwapTensor>(ENV, cv.t2_info.dims.at(0), cv.t2_info.dims.at(1));

      REQUIRE_THROWS(Tensor::contract(std::move(t_sden_u), std::move(t_swap_u), cv.wires));
    }
  }

  SECTION("identity-dense") {
    // Valid identities
    for (auto& cv : gen::id_vals) {
      auto t_sden_u = std::make_unique<DDenseTensor>(ENV, cv.t1_info.dims, cv.t1_info.els, 0);
      auto t_id_u = std::make_unique<IdentityTensor>(ENV, utils::split_dims(cv.t2_info.dims, cv.t2_info.dims.size() / 2).first);

      auto t_r1_u = Tensor::contract(std::move(t_sden_u), std::move(t_id_u), cv.wires);

      qtnh::tidx_tup t_r1_dims = cv.t3_info.dims;
      std::vector<qtnh::tel> t_r1_els = cv.t3_info.els;

      REQUIRE(t_r1_u->getDims() == t_r1_dims);
      TIndexing ti_r1(t_r1_dims);
      for (auto idxs : ti_r1) {
        auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_dims));
        REQUIRE(eq(t_r1_u->getLocEl(idxs).value(), el));
      }
    }
  }

  SECTION("convert-dense") {
    // The only convert contractions that can be tested serially are tensor products
    // Also test for invalid convert contractions

    qtnh::tidx_tup dims { 2, 2 };
    std::vector<qtnh::tel> els { 1.0i, 2.0i, 3.0i, 4.0i };
    auto t_sden_u = std::make_unique<SDenseTensor>(ENV, dims, els);
    auto t_dden_u = std::make_unique<DDenseTensor>(ENV, dims, els, 0);
    auto t_conv1_u = std::make_unique<ConvertTensor>(ENV, qtnh::tidx_tup {});
    auto t_conv2_u = std::make_unique<ConvertTensor>(ENV, qtnh::tidx_tup {});

    // SDenseTensor to DDenseTensor
    auto t_r1_u = Tensor::contract(std::move(t_sden_u), std::move(t_conv1_u), {});
    REQUIRE(dynamic_cast<SDenseTensor*>(t_r1_u.get()) == nullptr);
    REQUIRE(dynamic_cast<DDenseTensor*>(t_r1_u.get()) != nullptr);

    REQUIRE(t_r1_u->getDims() == dims);
    TIndexing ti_r1(dims);
    for (auto idxs : ti_r1) {
      auto el = els.at(utils::idxs_to_i(idxs, dims));
      REQUIRE(eq(t_r1_u->getLocEl(idxs).value(), el));
    }

    // DDenseTensor to SDenseTensor
    auto t_r2_u = Tensor::contract(std::move(t_dden_u), std::move(t_conv2_u), {});
    REQUIRE(dynamic_cast<SDenseTensor*>(t_r2_u.get()) != nullptr);
    REQUIRE(dynamic_cast<DDenseTensor*>(t_r2_u.get()) == nullptr);

    REQUIRE(t_r2_u->getDims() == dims);
    TIndexing ti_r2(dims);
    for (auto idxs : ti_r2) {
      auto el = els.at(utils::idxs_to_i(idxs, dims));
      REQUIRE(eq(t_r2_u->getLocEl(idxs).value(), el));
    }
  }
}
