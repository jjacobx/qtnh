#include <catch2/catch_test_macros.hpp>
#include "tensor/indexing.hpp"

TEST_CASE("tensor-indexing-setup") {
  using qtnh::TIndexing;
  using qtnh::TIFlag;

  qtnh::tidx_tup dims = { 2, 3, 4, 5 };
  std::vector<TIFlag> ifls(4, { "open", 0 });
  ifls.at(1) =  { "closed", 0 };

  SECTION("construction") {
    REQUIRE_NOTHROW(TIndexing(dims));
    REQUIRE_NOTHROW(TIndexing(dims, ifls));

    // No checking implemented yet. 
    // std::vector<TIFlag> invalid_ifls(5, { "open", 0 });
    // REQUIRE_THROWS(TIndexing(dims, invalid_ifls));
  }

  TIndexing ti1(dims);
  TIndexing ti2(dims, ifls);

  std::vector<TIFlag> ifls_def(4, { "default", 0 });

  SECTION("accessors") {
    REQUIRE(ti1.dims() == dims);
    REQUIRE(ti1.ifls() == ifls_def);
    REQUIRE(ti2.ifls() == ifls);
  }
}

TEST_CASE("tensor-indexing-iteration") {
  using qtnh::TIndexing;
  using qtnh::TIFlag;

  TIndexing ti({ 2, 2, 2, 2 }, {{ "default", 0 }, { "default", 0 }, { "other", 0 }, { "default", 0 }});

  SECTION("next-indices") {
    REQUIRE(ti.next({ 0, 0, 0, 0 }) == qtnh::tidx_tup { 0, 0, 0, 1 });
    REQUIRE(ti.next({ 0, 0, 0, 1 }) == qtnh::tidx_tup { 0, 1, 0, 0 });
    REQUIRE(ti.next({ 0, 0, 0, 0 }, "other") == qtnh::tidx_tup { 0, 0, 1, 0 });

    REQUIRE_THROWS(ti.next({ 1, 1, 1, 1 }));
  }

  SECTION("prev-indices") {
    REQUIRE_THROWS(ti.prev({ 0, 0, 0, 0 }));

    REQUIRE(ti.prev({ 0, 0, 0, 1 }) == qtnh::tidx_tup { 0, 0, 0, 0 });
    REQUIRE(ti.prev({ 0, 1, 0, 0 }) == qtnh::tidx_tup { 0, 0, 0, 1 });
    REQUIRE(ti.prev({ 0, 0, 1, 0 }, "other") == qtnh::tidx_tup { 0, 0, 0, 0 });
  }

  SECTION("iterate-indices") {
    qtnh::tidx_tup idxs2 = { 0, 0, 0, 0 };

    for (auto idxs1 : ti.tup()) {
      REQUIRE(idxs1 == idxs2);

      if (!ti.isLast(idxs2)) {
        idxs2 = ti.next(idxs2);
      }
    }
  }
}

TEST_CASE("tensor-indexing-operations") {
  using qtnh::TIndexing;
  using qtnh::TIFlag;

  TIndexing ti1({ 2, 3, 4, 5 },  {{ "default", 0 }, { "default", 0 }, { "other", 0 }, { "default", 0 }});

  SECTION("cut-type") {
    TIndexing ti11 = ti1.cut();
    TIndexing ti12 = ti1.cut("other");

    REQUIRE(ti11.dims() == qtnh::tidx_tup { 4 });
    REQUIRE(ti12.dims() == qtnh::tidx_tup { 2, 3, 5 });
  }

  TIndexing ti2({ 6, 7, 8 });

  SECTION("append-indexings") {
    TIndexing ti3 = TIndexing::app(ti1, ti2);
    REQUIRE(ti3.dims() == qtnh::tidx_tup { 2, 3, 4, 5, 6, 7, 8 });
  }
}
