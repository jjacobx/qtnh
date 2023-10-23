#include <catch2/catch_test_macros.hpp>
#include "tensor.hpp"

using namespace std::complex_literals;

TEST_CASE("create-tensor-test") {
    tels_array x_els = { 0.0, 1.0, 1.0, 0.0 };
    tels_array y_els = { 0.0, -1.0i, 1.0i, 0.0 };
    tels_array copy_els = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

    tidx_tuple x_dims = { 2, 2 };
    tidx_tuple y_dims = { 2, 2 };
    tidx_tuple copy_dims = { 2, 2, 2 };

    Tensor* x;
    Tensor* y;
    Tensor* copy;

    REQUIRE_NOTHROW(x = new Tensor(x_dims, x_els));
    REQUIRE_NOTHROW(y = new Tensor(y_dims, y_els));
    REQUIRE_NOTHROW(copy = new Tensor(copy_dims, copy_els));
}


TEST_CASE("split-tensor-test") {
    tels_array copy_els = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
    tidx_tuple copy_dims = { 2, 2, 2 };

    Tensor copy(copy_dims, copy_els);
    std::vector<Tensor> ts;

    REQUIRE_NOTHROW(ts = copy.split(1));
}
