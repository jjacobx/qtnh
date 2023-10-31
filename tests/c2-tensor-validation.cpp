#include <catch2/catch_test_macros.hpp>
#include "tensor.hpp"

using namespace std::complex_literals;

TEST_CASE("create-tensor-validation") {
    tidx_tuple t1_dims = { 2, 2 };
    tels_array t1_els = { 0.0, 0.0, 0.0, 0.0 };
    Tensor t11(t1_dims);
    Tensor t12(t1_dims, t1_els);

    tidx_tuple t2_dims = { 1 };
    tels_array t2_els = { 0.0 };
    Tensor t21;
    Tensor t22(t2_dims, t2_els);

    REQUIRE(t11 == t12);
    REQUIRE(t21 == t22);
    REQUIRE(t11 != t21);
    REQUIRE(t12 != t22);

    REQUIRE(t11.getID() == 1);
    REQUIRE(t12.getID() == 2);
    REQUIRE(t21.getID() == 3);
    REQUIRE(t22.getID() == 4);
}

TEST_CASE("access-tensor-validation") {
    tels_array t1_els = { 0.0, 1.0, 2.0, 3.0 };
    tidx_tuple t1_dims = { 2, 2 };
    Tensor t1(t1_dims, t1_els);

    tels_array t2_els = { 0.0, 2.0, 1.0, 3.0 };
    tidx_tuple t2_dims = { 2, 2 };
    Tensor t2(t2_dims, t2_els);

    tidx_tuple i1 = { 0, 1 };
    tidx_tuple i2 = { 1, 0 };
    complex c = t1[i2];
    t1[i2] = t1[i1];
    t1[i1] = c;

    REQUIRE(t1 == t2);
}

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


TEST_CASE("contract-tensor-network-validation") {
    tidx_tuple t1_dims = { 2, 2, 2 };
    tels_array t1_els = { 0.0 + 1.0i, 1.0 + 0.0i, 1.0 + 0.0i, 0.0 + 1.0i, 1.0 + 0.0i, 0.0 + 1.0i, 0.0 + 1.0i, 1.0 + 0.0i };
    Tensor t1(t1_dims, t1_els);

    tidx_tuple t2_dims = { 2, 4 };
    tels_array t2_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 1.0 - 1.0i, 2.0 - 2.0i, 3.0 - 3.0i, 4.0 - 4.0i };
    Tensor t2(t2_dims, t2_els);

    tidx_tuple tr_dims = { 2, 2, 4 };
    tels_array tr_els = { 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 
                          2.0 + 2.0i, 4.0 + 4.0i, 6.0 + 6.0i, 8.0 + 8.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i, 0.0 + 0.0i };
    Tensor tr(tr_dims, tr_els);

    std::pair<Tensor, Tensor> b_tensors(t1, t2);
    std::pair<int, int> b_dims(1, 0);
    Bond b(b_tensors, b_dims);

    std::vector<Tensor> tn_tensors = { t1, t2 };
    std::vector<Bond> tn_bonds = { b };
    TensorNetwork tn(tn_tensors, tn_bonds);
    tn.contract();

    REQUIRE(tr == tn.getTensor(0));
}
