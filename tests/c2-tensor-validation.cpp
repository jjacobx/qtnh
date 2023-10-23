#include <catch2/catch_test_macros.hpp>
#include "tensor.hpp"

using namespace std::complex_literals;

TEST_CASE("split-tensor-validation") {
    tidx_tuple t0_dims = { 3, 2, 2 };
    tels_array t0_els = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0 };

    Tensor copy(t0_dims, t0_els);
    auto ts = copy.split(1);
    Tensor t1 = ts.at(0);
    Tensor t2 = ts.at(1);
    
    tidx_tuple t1_dims = { 3, 2 };
    tidx_tuple t2_dims = { 3, 2 };
    tels_array t1_els = { 0.0, 1.0, 4.0, 5.0, 8.0, 9.0 };
    tels_array t2_els = { 2.0, 3.0, 6.0, 7.0, 10.0, 11.0 };

    Tensor t1_res(t1_dims, t1_els);
    Tensor t2_res(t2_dims, t2_els);

    REQUIRE(t1 == t1_res);
    REQUIRE(t2 == t2_res);
}
