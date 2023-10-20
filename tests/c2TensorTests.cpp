#include <catch2/catch_test_macros.hpp>
#include "tensor.hpp"

using namespace std::complex_literals;

TEST_CASE("Tensor") {
    std::vector<complex> x_els = { 0.0, 1.0, 1.0, 0.0 };
    std::vector<complex> y_els = { 0.0, -1.0i, 1.0i, 0.0 };
    std::vector<complex> copy_els = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

    tidx_tuple x_dims = { 2, 2 };
    tidx_tuple y_dims = { 2, 2 };
    tidx_tuple copy_dims = { 2, 2, 2 };

    Tensor * x;
    Tensor * y;
    Tensor * copy;

    REQUIRE_NOTHROW(x = new Tensor(x_dims, x_els));
    REQUIRE_NOTHROW(y = new Tensor(y_dims, y_els));
    REQUIRE_NOTHROW(copy = new Tensor(copy_dims, copy_els));
}
