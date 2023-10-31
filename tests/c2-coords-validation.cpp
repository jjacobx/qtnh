#include <catch2/catch_test_macros.hpp>
#include "coords.hpp"

TEST_CASE("create-indexing-validation") {
    tidx_tuple dims = { 2, 3, 4, 5 };
    std::size_t closed_idx = 1;
    tidx_flags flags1(4, TIFlag::open);
    tidx_flags flags2 = flags1;
    flags2.at(1) = TIFlag::closed;

    TIndexing ti1(dims);
    TIndexing ti2(dims, flags1);
    TIndexing ti3(dims, closed_idx);
    TIndexing ti4(dims, flags2);

    REQUIRE(ti1 == ti2);
    REQUIRE(ti3 == ti4);
    REQUIRE(ti1 != ti3);
    REQUIRE(ti2 != ti4);
}

TEST_CASE("increase-index-validation") {
    tidx_tuple dims = { 5, 4, 3, 2 };
    tidx_flags flags(4, TIFlag::open);
    flags.at(2) = TIFlag::closed;
    
    TIndexing ti(dims, flags);

    tidx_tuple idx = { 0, 0, 0, 0 };
    tidx_tuple target_idx1 = { 0, 0, 0, 1 };
    tidx_tuple target_idx2 = { 0, 1, 0, 0 };

    idx = ti.next(idx);
    REQUIRE(idx == target_idx1);

    idx = ti.next(idx);
    REQUIRE(idx == target_idx2);
}

TEST_CASE("decrease-index-validation") {
    tidx_tuple dims = { 5, 4, 3, 2 };
    tidx_flags flags(4, TIFlag::open);
    flags.at(2) = TIFlag::closed;
    
    TIndexing ti(dims, flags);

    tidx_tuple idx = { 4, 3, 2, 1 };
    tidx_tuple target_idx1 = { 4, 3, 2, 0 };
    tidx_tuple target_idx2 = { 4, 2, 2, 1 };

    idx = ti.prev(idx);
    REQUIRE(idx == target_idx1);

    idx = ti.prev(idx);
    REQUIRE(idx == target_idx2);
}

TEST_CASE("iterate-indexing-validation") {
    tidx_tuple dims = { 2, 3, 4, 5 };
    tidx_flags flags(4, TIFlag::open);
    flags.at(1) = TIFlag::closed;

    TIndexing ti(dims, flags);
    tidx_tuple j = { 0, 0, 0, 0 };

    for (auto i : ti) {
        REQUIRE(i == j);

        if (!ti.isLast(j)) {
            j = ti.next(j);
        }
    }
}

TEST_CASE("cut-indexing-validation") {
    tidx_tuple dims1 = { 2, 3, 4, 5 };
    tidx_flags flags1(4, TIFlag::open);
    flags1.at(1) = TIFlag::closed;
    TIndexing ti1(dims1, flags1);

    tidx_tuple dims2 = { 2, 4, 5 };
    tidx_flags flags2(3, TIFlag::open);
    TIndexing ti2(dims2, flags2);
    
    REQUIRE(ti2 == ti1.cut(TIFlag::closed));
}

TEST_CASE("append-indexings-validation") {
    tidx_tuple dims1 = { 2, 3, 4 };
    tidx_flags flags1 = { TIFlag::open, TIFlag::closed, TIFlag::open };
    tidx_tuple dims2 = { 3, 2 };
    tidx_flags flags2 = { TIFlag::closed, TIFlag::open };

    tidx_tuple dims12 = { 2, 3, 4, 3, 2 };
    tidx_flags flags12 = { TIFlag::open, TIFlag::closed, TIFlag::open, TIFlag::closed, TIFlag::open };

    TIndexing ti1(dims1, flags1);
    TIndexing ti2(dims2, flags2);
    TIndexing ti12(dims12, flags12);

    REQUIRE(ti12 == TIndexing::app(ti1, ti2));
}
