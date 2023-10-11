#include <catch2/catch_test_macros.hpp>
#include "tensor.hpp"

using namespace std::complex_literals;

TEST_CASE("Tensor") {
    std::vector<complex> x_els = { 0.0, 1.0, 1.0, 0.0 };
    std::vector<complex> y_els = { 0.0, -1.0i, 1.0i, 0.0 };
    std::vector<complex> copy_els = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };

    std::vector<int> x_dims = { 2, 2 };
    std::vector<int> y_dims = { 2, 2 };
    std::vector<int> copy_dims = { 2, 2, 2 };

    Tensor * x;
    Tensor * y;
    Tensor * copy;

    REQUIRE_NOTHROW(x = new Tensor(x_els, x_dims));
    REQUIRE_NOTHROW(y = new Tensor(y_els, y_dims));
    REQUIRE_NOTHROW(copy = new Tensor(copy_els, copy_dims));
}
