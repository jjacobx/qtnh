#include <catch2/catch_test_macros.hpp>
#include <memory>
#include <vector>

#include "tensor/dense.hpp"
#include "tensor/special.hpp"

qtnh::QTNHEnv ENV;

TEST_CASE("tensor-construction") {
  using namespace qtnh;
  using namespace std::complex_literals;

  SECTION("swap-tensor") {
    auto tu = std::make_unique<SwapTensor>(ENV, 2, 2);
  }

  SECTION("identity-tensor") {
    auto tu = std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 2, 2 });
  }

  SECTION("convert-tensor") {
    auto tu = std::make_unique<IdentityTensor>(ENV, qtnh::tidx_tup { 2, 2 });
  }

  std::vector<qtnh::tel> els { 1.0i, 2.0i, 3.0i, 4.0i };

  SECTION("shared-dense-tensor") {
    auto t1u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els);
    auto t2u = std::make_unique<SDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els, false);
  }

  SECTION("distributed-dense-tensor") {
    auto t1u = std::make_unique<DDenseTensor>(ENV, qtnh::tidx_tup { 2, 2 }, els, 0);
  }
}
