#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "core/utils.hpp"
#include "tensor/dense.hpp"
#include "tensor/indexing.hpp"
#include "tensor/special.hpp"

using namespace qtnh;
using namespace std::complex_literals;

QTNHEnv ENV;

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

  std::vector<qtnh::tel> els { 1.0i, 2.0i, 3.0i, 4.0i };
  auto t_sden_u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els);
  auto t_dden_u = std::make_unique<DDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els, 0);

  SECTION("dense-dense") {
    auto t_sden1_u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els);
    auto t_sden2_u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els);

    auto t_r1_u = Tensor::contract(std::move(t_sden1_u), std::move(t_sden2_u), {{ 0, 1 }});
    std::vector<qtnh::tel> t_r1_els { -7.0,  -15.0, -10.0, -22.0 };

    TIndexing ti_r1(t_r1_u->getDims());
    for (auto idxs : ti_r1) {
      auto el = t_r1_els.at(utils::idxs_to_i(idxs, t_r1_u->getDims()));
      REQUIRE(t_r1_u->getLocEl(idxs).value() == el);
    }
  }

  SECTION("swap-dense") {

  }

  SECTION("identity-dense") {

  }

  SECTION("convert-dense") {

  }
}
