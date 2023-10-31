#include <catch2/catch_test_macros.hpp>
#include "coords.hpp"

TEST_CASE("create-indexing-test") {
    tidx_tuple dims = { 2, 3, 4, 5 };
    std::size_t closed_idx = 1;
    tidx_flags flags(4, TIFlag::open);
    flags.at(1) = TIFlag::closed;
    
    TIndexing* ti;

    REQUIRE_NOTHROW(ti = new TIndexing(dims));
    REQUIRE_NOTHROW(ti = new TIndexing(dims, closed_idx));
    REQUIRE_NOTHROW(ti = new TIndexing(dims, flags));

    tidx_flags invalid_flags(5, TIFlag::open);

    REQUIRE_THROWS(ti = new TIndexing(dims, invalid_flags));
}

TEST_CASE("increase-index-test") {
    tidx_tuple dims = { 2, 3, 4, 5 };
    TIndexing ti(dims);

    tidx_tuple start_idx = { 0, 0, 0, 0 };
    REQUIRE_NOTHROW(start_idx = ti.next(start_idx));

    tidx_tuple end_idx = { 1, 2, 3, 4 };
    REQUIRE_THROWS(end_idx = ti.next(end_idx));
}

TEST_CASE("decrease-index-test") {
    tidx_tuple dims = { 2, 3, 4, 5 };
    TIndexing ti(dims);

    tidx_tuple end_idx = { 1, 2, 3, 4 };
    REQUIRE_NOTHROW(end_idx = ti.prev(end_idx));

    tidx_tuple start_idx = { 0, 0, 0, 0 };
    REQUIRE_THROWS(start_idx = ti.prev(start_idx));
}

TEST_CASE("iterate-indexing-test") {
    tidx_tuple dims = { 2, 3, 4, 5 };
    TIndexing ti(dims);

    REQUIRE_NOTHROW([&](){
        for (auto i : ti);
    }());
}

TEST_CASE("cut-indexing-test") {
    tidx_tuple dims1 = { 2, 3, 4, 5 };
    tidx_flags flags1(4, TIFlag::open);
    flags1.at(1) = TIFlag::closed;

    TIndexing ti1(dims1, flags1);
    TIndexing ti2;
    
    REQUIRE_NOTHROW(ti2 = ti1.cut(TIFlag::closed));
}

TEST_CASE("append-indexings-test") {
    tidx_tuple dims1 = { 2, 3, 4, 5 };
    tidx_tuple dims2 = { 5, 4, 3, 2 };
    TIndexing ti1(dims1);
    TIndexing ti2(dims2);
    
    TIndexing ti12;
    REQUIRE_NOTHROW(ti12 = TIndexing::app(ti1, ti2));
}
