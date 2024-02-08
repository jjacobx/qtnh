#include <catch2/catch_test_macros.hpp>
#include "indexing.hpp"

TEST_CASE("create-indexing-validation") {
    qtnh::tidx_tup dims = { 2, 3, 4, 5 };
    std::size_t closed_idx = 1;
    qtnh::tidx_flags flags1(4, qtnh::TIdxFlag::open);
    qtnh::tidx_flags flags2 = flags1;
    flags2.at(1) = qtnh::TIdxFlag::closed;

    qtnh::TIndexing ti1(dims);
    qtnh::TIndexing ti2(dims, flags1);
    qtnh::TIndexing ti3(dims, closed_idx);
    qtnh::TIndexing ti4(dims, flags2);

    REQUIRE(ti1 == ti2);
    REQUIRE(ti3 == ti4);
    REQUIRE(ti1 != ti3);
    REQUIRE(ti2 != ti4);
}

TEST_CASE("increase-index-validation") {
    qtnh::tidx_tup dims = { 5, 4, 3, 2 };
    qtnh::tidx_flags flags(4, qtnh::TIdxFlag::open);
    flags.at(2) = qtnh::TIdxFlag::closed;
    
    qtnh::TIndexing ti(dims, flags);

    qtnh::tidx_tup idx = { 0, 0, 0, 0 };
    qtnh::tidx_tup target_idx1 = { 0, 0, 0, 1 };
    qtnh::tidx_tup target_idx2 = { 0, 1, 0, 0 };

    idx = ti.next(idx);
    REQUIRE(idx == target_idx1);

    idx = ti.next(idx);
    REQUIRE(idx == target_idx2);
}

TEST_CASE("decrease-index-validation") {
    qtnh::tidx_tup dims = { 5, 4, 3, 2 };
    qtnh::tidx_flags flags(4, qtnh::TIdxFlag::open);
    flags.at(2) = qtnh::TIdxFlag::closed;
    
    qtnh::TIndexing ti(dims, flags);

    qtnh::tidx_tup idx = { 4, 3, 2, 1 };
    qtnh::tidx_tup target_idx1 = { 4, 3, 2, 0 };
    qtnh::tidx_tup target_idx2 = { 4, 2, 2, 1 };

    idx = ti.prev(idx);
    REQUIRE(idx == target_idx1);

    idx = ti.prev(idx);
    REQUIRE(idx == target_idx2);
}

TEST_CASE("iterate-indexing-validation") {
    qtnh::tidx_tup dims = { 2, 3, 4, 5 };
    qtnh::tidx_flags flags(4, qtnh::TIdxFlag::open);
    flags.at(1) = qtnh::TIdxFlag::closed;

    qtnh::TIndexing ti(dims, flags);
    qtnh::tidx_tup j = { 0, 0, 0, 0 };

    for (auto i : ti) {
        REQUIRE(i == j);

        if (!ti.isLast(j)) {
            j = ti.next(j);
        }
    }
}

TEST_CASE("cut-indexing-validation") {
    qtnh::tidx_tup dims1 = { 2, 3, 4, 5 };
    qtnh::tidx_flags flags1(4, qtnh::TIdxFlag::open);
    flags1.at(1) = qtnh::TIdxFlag::closed;
    qtnh::TIndexing ti1(dims1, flags1);

    qtnh::tidx_tup dims2 = { 2, 4, 5 };
    qtnh::tidx_flags flags2(3, qtnh::TIdxFlag::open);
    qtnh::TIndexing ti2(dims2, flags2);
    
    REQUIRE(ti2 == ti1.cut(qtnh::TIdxFlag::closed));
}

TEST_CASE("append-indexings-validation") {
    qtnh::tidx_tup dims1 = { 2, 3, 4 };
    qtnh::tidx_flags flags1 = { qtnh::TIdxFlag::open, qtnh::TIdxFlag::closed, qtnh::TIdxFlag::open };
    qtnh::tidx_tup dims2 = { 3, 2 };
    qtnh::tidx_flags flags2 = { qtnh::TIdxFlag::closed, qtnh::TIdxFlag::open };

    qtnh::tidx_tup dims12 = { 2, 3, 4, 3, 2 };
    qtnh::tidx_flags flags12 = { qtnh::TIdxFlag::open, qtnh::TIdxFlag::closed, qtnh::TIdxFlag::open, qtnh::TIdxFlag::closed, qtnh::TIdxFlag::open };

    qtnh::TIndexing ti1(dims1, flags1);
    qtnh::TIndexing ti2(dims2, flags2);
    qtnh::TIndexing ti12(dims12, flags12);

    REQUIRE(ti12 == qtnh::TIndexing::app(ti1, ti2));
}
