#include <catch2/catch_test_macros.hpp>
#include "tensor/indexing.hpp"

TEST_CASE("create-indexing-test") {
    qtnh::tidx_tup dims = { 2, 3, 4, 5 };
    std::size_t closed_idx = 1;
    qtnh::tidx_flags flags(4, qtnh::TIdxFlag::open);
    flags.at(1) = qtnh::TIdxFlag::closed;
    
    qtnh::TIndexing* ti;

    REQUIRE_NOTHROW(ti = new qtnh::TIndexing(dims));
    REQUIRE_NOTHROW(ti = new qtnh::TIndexing(dims, closed_idx));
    REQUIRE_NOTHROW(ti = new qtnh::TIndexing(dims, flags));

    qtnh::tidx_flags invalid_flags(5, qtnh::TIdxFlag::open);

    REQUIRE_THROWS(ti = new qtnh::TIndexing(dims, invalid_flags));
}

TEST_CASE("increase-index-test") {
    qtnh::tidx_tup dims = { 2, 3, 4, 5 };
    qtnh::TIndexing ti(dims);

    qtnh::tidx_tup start_idx = { 0, 0, 0, 0 };
    REQUIRE_NOTHROW(start_idx = ti.next(start_idx));

    qtnh::tidx_tup end_idx = { 1, 2, 3, 4 };
    REQUIRE_THROWS(end_idx = ti.next(end_idx));
}

TEST_CASE("decrease-index-test") {
    qtnh::tidx_tup dims = { 2, 3, 4, 5 };
    qtnh::TIndexing ti(dims);

    qtnh::tidx_tup end_idx = { 1, 2, 3, 4 };
    REQUIRE_NOTHROW(end_idx = ti.prev(end_idx));

    qtnh::tidx_tup start_idx = { 0, 0, 0, 0 };
    REQUIRE_THROWS(start_idx = ti.prev(start_idx));
}

TEST_CASE("iterate-indexing-test") {
    qtnh::tidx_tup dims = { 2, 3, 4, 5 };
    qtnh::TIndexing ti(dims);

    REQUIRE_NOTHROW([&](){
        for (auto i : ti);
    }());
}

TEST_CASE("cut-indexing-test") {
    qtnh::tidx_tup dims1 = { 2, 3, 4, 5 };
    qtnh::tidx_flags flags1(4, qtnh::TIdxFlag::open);
    flags1.at(1) = qtnh::TIdxFlag::closed;

    qtnh::TIndexing ti1(dims1, flags1);
    qtnh::TIndexing ti2;
    
    REQUIRE_NOTHROW(ti2 = ti1.cut(qtnh::TIdxFlag::closed));
}

TEST_CASE("append-indexings-test") {
    qtnh::tidx_tup dims1 = { 2, 3, 4, 5 };
    qtnh::tidx_tup dims2 = { 5, 4, 3, 2 };
    qtnh::TIndexing ti1(dims1);
    qtnh::TIndexing ti2(dims2);
    
    qtnh::TIndexing ti12;
    REQUIRE_NOTHROW(ti12 = qtnh::TIndexing::app(ti1, ti2));
}
