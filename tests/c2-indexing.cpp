#include <catch2/catch_test_macros.hpp>
#include "tensor/indexing.hpp"

TEST_CASE("tensor-indexing-setup") {
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

TEST_CASE("tensor-indexing-iteration") {
  using qtnh::TIndexing;
  using qtnh::TIdxT;

  TIndexing ti({ 2, 2, 2, 2 }, 2);

  SECTION("next-indices") {
    REQUIRE(ti.next({ 0, 0, 0, 0 }) == qtnh::tidx_tup { 0, 0, 0, 1 });
    REQUIRE(ti.next({ 0, 0, 0, 1 }) == qtnh::tidx_tup { 0, 1, 0, 0 });
    REQUIRE(ti.next({ 0, 0, 0, 0 }, TIdxT::closed) == qtnh::tidx_tup { 0, 0, 1, 0 });

    REQUIRE_THROWS(ti.next({ 1, 1, 1, 1 }));
  }

  SECTION("prev-indices") {
    REQUIRE_THROWS(ti.prev({ 0, 0, 0, 0 }));

    REQUIRE(ti.prev({ 0, 0, 0, 1 }) == qtnh::tidx_tup { 0, 0, 0, 0 });
    REQUIRE(ti.prev({ 0, 1, 0, 0 }) == qtnh::tidx_tup { 0, 0, 0, 1 });
    REQUIRE(ti.prev({ 0, 0, 1, 0 }, TIdxT::closed) == qtnh::tidx_tup { 0, 0, 0, 0 });
  }

  SECTION("iterate-indices") {
    qtnh::tidx_tup idxs2 = { 0, 0, 0, 0 };

    for (auto idxs1 : ti) {
      REQUIRE(idxs1 == idxs2);

      if (!ti.isLast(idxs2)) {
        idxs2 = ti.next(idxs2);
      }
    }
  }
}

TEST_CASE("tensor-indexing-operations") {
  using qtnh::TIndexing;
  using qtnh::TIdxT;

  TIndexing ti1({ 2, 3, 4, 5 }, 2);

  SECTION("cut-type") {
    TIndexing ti12 = ti1.cut(TIdxT::closed);
    TIndexing ti13 = ti1.cut(TIdxT::open);

    REQUIRE(ti12.getDims() == qtnh::tidx_tup { 2, 3, 5 });
    REQUIRE(ti13.getDims() == qtnh::tidx_tup { 4 });
  }

  TIndexing ti2({ 6, 7, 8 });

  SECTION("append-indexings") {
    TIndexing ti3 = TIndexing::app(ti1, ti2);
    REQUIRE(ti3.getDims() == qtnh::tidx_tup { 2, 3, 4, 5, 6, 7, 8 });
  }
}
