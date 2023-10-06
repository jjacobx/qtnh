#include <iostream>

#include "tensor.hpp"

using namespace std::complex_literals;

int main() {
    std::vector<complex> t1_els = { 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i, 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i };
    std::vector<complex> t2_els = { 5.0 + 5.0i, 6.0 + 6.0i, 7.0 + 7.0i, 8.0 + 8.0i, 1.0 + 1.0i, 2.0 + 2.0i, 3.0 + 3.0i, 4.0 + 4.0i };

    std::vector<int> t1_dims = { 2, 2, 2 };
    std::vector<int> t2_dims = { 4, 2 };

    Tensor t1(t1_els, t1_dims);
    Tensor t2(t2_els, t2_dims);

    std::pair<Tensor, Tensor> b1_tensors(t1, t2);
    std::pair<int, int> b1_dims(2, 1);

    Bond b1(b1_tensors, b1_dims);

    std::vector<Tensor> tn1_tensors = { t1, t2 };
    std::vector<Bond> tn1_bonds = { b1 };

    TensorNetwork tn1(tn1_tensors, tn1_bonds);

    Tensor result = tn1.contract();

    std::vector<int> result_coords = { 0 };

    std::cout << "Tout[0] = " << result.getEl(result_coords) << std::endl;

    return 0;
}
