#include <catch2/catch_test_macros.hpp>
#include "tensor/indexing.hpp"

TEST_CASE("tensor-indexing") {
  using qtnh::TIndexing;
  using qtnh::TIdxT;

  qtnh::tidx_tup dims = { 2, 3, 4, 5 };
  std::size_t closed_idx = 1;
  qtnh::tifl_tup ifls(4, { TIdxT::open, 0 });
  ifls.at(1) =  { TIdxT::closed, 0 };

  SECTION("construction") {
    REQUIRE_NOTHROW(TIndexing(dims));
    REQUIRE_NOTHROW(TIndexing(dims, closed_idx));
    REQUIRE_NOTHROW(TIndexing(dims, ifls));

    qtnh::tifl_tup invalid_ifls(5, { qtnh::TIdxT::open, 0 });
    REQUIRE_THROWS(TIndexing(dims, invalid_ifls));
  }

  TIndexing ti1(dims);
  TIndexing ti2(dims, 1);
  TIndexing ti3(dims, ifls);

  SECTION("accessors") {
    REQUIRE(ti1.getDims() == dims);
    REQUIRE(ti3.getIFls() == ifls);
  }

  SECTION("comparison") {
    REQUIRE(ti1 != ti2);
    REQUIRE(ti2 == ti3);
  }
}