#include <iostream>

#include "tensor.hpp"

using namespace std::complex_literals;

int main() {
    std::vector<complex> t1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
    std::vector<complex> t2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

    dim_tuple t1_dims = { 2, 2, 2 };
    dim_tuple t2_dims = { 4, 2 };

    Tensor t1(t1_dims, t1_els);
    Tensor t2(t2_dims, t2_els);

    dim_tuple dims = t1.getDims();

    std::pair<Tensor, Tensor> b1_tensors(t1, t2);
    std::pair<int, int> b1_dims(2, 1);

    Bond b1(b1_tensors, b1_dims);

    std::vector<Tensor> tn1_tensors = { t1, t2 };
    std::vector<Bond> tn1_bonds = { b1 };

    TensorNetwork tn1(tn1_tensors, tn1_bonds);

    tn1.contract();
    Tensor result = tn1.getTensor(0);

    dim_tuple result_coords = { 0, 0, 0 };

    std::cout << "Tout[0] = " << result.getEl(result_coords) << std::endl;

    dim_tuple limits = { 2, 4, 3 };
    dim_flags flags = { DimFlag::variable, DimFlag::fixed, DimFlag::variable};
    dim_tuple current = { 0, 0, 0 };
    Coordinates c1(limits, 1);
    c1.next();
    dim_tuple curr = c1.getCurrent();

    std::cout << "c1 = " << c1 << std::endl;
    std::cout << "c1 can increase: " << c1.canIncrease() << std::endl;
    std::cout << "c1 can decrease: " << c1.canDecrease() << std::endl;

    Coordinates c2 = c1.complement();
    std::cout << "c2 = " << c2 << std::endl;
    c2.next();
    std::cout << "c2 = " << c2 << std::endl;

    Coordinates c3 = c1.dropFixed();
    std::cout << "c3 = " << c3 << std::endl;

    std::cout << "c1 && c3 = " << (c1 && c3) << std::endl;

    return 0;
}
