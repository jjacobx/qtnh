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

TEST_CASE("access-tensor-test") {
    tels_array t_els = { 0.0, 1.0, 2.0, 3.0 };
    tidx_tuple t_dims = { 2, 2 };
    Tensor t(t_dims, t_els);

    tidx_tuple i1 = { 0, 1 };
    tidx_tuple i2 = { 1, 0 };
    complex c;

    REQUIRE_NOTHROW(c = t[i1]);
    REQUIRE_NOTHROW(t[i2] = c);
}

TEST_CASE("split-tensor-test") {
    tels_array copy_els = { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 };
    tidx_tuple copy_dims = { 2, 2, 2 };

    Tensor copy(copy_dims, copy_els);
    std::vector<Tensor> ts;

    REQUIRE_NOTHROW(ts = copy.split(1));
}

TEST_CASE("contract-tensor-network-test") {
    tidx_tuple t1_dims = { 2, 2, 2 };
    tels_array t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
    Tensor t1(t1_dims, t1_els);

    tidx_tuple t2_dims = { 2, 4 };
    tels_array t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
    Tensor t2(t2_dims, t2_els);

    std::pair<Tensor, Tensor> b_tensors(t1, t2);
    std::pair<int, int> b_dims(1, 0);
    Bond b(b_tensors, b_dims);

    std::vector<Tensor> tn_tensors = { t1, t2 };
    std::vector<Bond> tn_bonds = { b };
    TensorNetwork tn(tn_tensors, tn_bonds);

    REQUIRE_NOTHROW(tn.contract());
}
