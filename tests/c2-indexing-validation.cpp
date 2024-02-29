#include <catch2/catch_test_macros.hpp>
#include "tensor/indexing.hpp"

TEST_CASE("create-indexing-validation") {
    qtnh::tidx_tup dims = { 2, 3, 4, 5 };
    std::size_t closed_idx = 1;
    qtnh::tifl_tup ifls1(4, { qtnh::TIdxT::open, 0 });
    qtnh::tifl_tup ifls2 = ifls1;
    ifls2.at(1) = { qtnh::TIdxT::closed, 0 };

    qtnh::TIndexing ti1(dims);
    qtnh::TIndexing ti2(dims, ifls1);
    qtnh::TIndexing ti3(dims, closed_idx);
    qtnh::TIndexing ti4(dims, ifls2);

    REQUIRE(ti1 == ti2);
    REQUIRE(ti3 == ti4);
    REQUIRE(ti1 != ti3);
    REQUIRE(ti2 != ti4);
}

TEST_CASE("increase-index-validation") {
    qtnh::tidx_tup dims = { 5, 4, 3, 2 };
    qtnh::tifl_tup ifls(4, { qtnh::TIdxT::open, 0 });
    ifls.at(2) = { qtnh::TIdxT::closed, 0 };
    
    qtnh::TIndexing ti(dims, ifls);

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
    qtnh::tifl_tup ifls(4, { qtnh::TIdxT::open, 0 });
    ifls.at(2) = { qtnh::TIdxT::closed, 1 };
    
    qtnh::TIndexing ti(dims, ifls);

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
    qtnh::tifl_tup ifls(4, { qtnh::TIdxT::open, 0 });
    ifls.at(1) = { qtnh::TIdxT::closed, 0 };

    qtnh::TIndexing ti(dims, ifls);
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
    qtnh::tifl_tup ifls1(4, { qtnh::TIdxT::open, 0 });
    ifls1.at(1) = { qtnh::TIdxT::closed, 0 };
    qtnh::TIndexing ti1(dims1, ifls1);

    qtnh::tidx_tup dims2 = { 2, 4, 5 };
    qtnh::tifl_tup ifls2(3, { qtnh::TIdxT::open, 0 });
    qtnh::TIndexing ti2(dims2, ifls2);
    
    REQUIRE(ti2 == ti1.cut(qtnh::TIdxT::closed));
}

TEST_CASE("append-indexings-validation") {
    qtnh::tidx_tup dims1 = { 2, 3, 4 };
    qtnh::tifl_tup ifls1 = { { qtnh::TIdxT::open, 0 }, { qtnh::TIdxT::closed, 0 }, { qtnh::TIdxT::open, 0 } };
    qtnh::tidx_tup dims2 = { 3, 2 };
    qtnh::tifl_tup ifls2 = { { qtnh::TIdxT::closed, 0 }, { qtnh::TIdxT::open, 0 }, };

    qtnh::tidx_tup dims12 = { 2, 3, 4, 3, 2 };
    qtnh::tifl_tup ifls12 = { { qtnh::TIdxT::open, 0 }, { qtnh::TIdxT::closed, 0 }, { qtnh::TIdxT::open, 0 }, { qtnh::TIdxT::closed, 0 }, { qtnh::TIdxT::open, 0 } };

    qtnh::TIndexing ti1(dims1, ifls1);
    qtnh::TIndexing ti2(dims2, ifls2);
    qtnh::TIndexing ti12(dims12, ifls12);

    REQUIRE(ti12 == qtnh::TIndexing::app(ti1, ti2));
}
