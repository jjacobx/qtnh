#include "tensor.hpp"

Tensor::Tensor() {
    std::vector<complex> data = { 0.0 };
    std::vector<int> dims = { 1 };
    
    this->data = data;
    this->dims = dims;
}

Tensor::Tensor(std::vector<complex> data, std::vector<int> dims) {
    this->data = data;
    this->dims = dims;
}

std::vector<int> Tensor::getDims() {
    return this->dims;
}

complex Tensor::getEl(std::vector<int> coords) {
    std::vector<int> dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    return this->data.at(idx);
}

Bond::Bond(std::pair<Tensor, Tensor> tensors, std::pair<int, int> dims) {
    this->tensors = tensors;
    this->dims = dims;
}

std::pair<Tensor, Tensor> Bond::getTensors() {
    return this->tensors;
}

std::pair<int, int> Bond::getDims() {
    return this->dims;
}

int Bond::getSize() {
    int dim = this->getDims().first;
    return this->tensors.first.getDims().at(dim);
}

TensorNetwork::TensorNetwork(std::vector<Tensor> tensors, std::vector<Bond> bonds) {
    this->tensors = tensors;
    this->bonds = bonds;
}

Tensor TensorNetwork::contract() {
    return Tensor();
}
