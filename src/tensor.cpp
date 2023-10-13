#include <cassert>
#include <iostream>
#include <stdexcept>

#include "tensor.hpp"

Coordinates::Coordinates(dim_tuple limits, std::size_t constant_dim) {
    dim_flags flags(limits.size(), DimFlag::variable);
    Coordinates(limits, flags);
}

Coordinates::Coordinates(dim_tuple limits, dim_flags flags) {
    dim_tuple current(limits.size(), 0);
    Coordinates(limits, flags, current);
}

Coordinates::Coordinates(dim_tuple limits, dim_flags flags, dim_tuple current) {
    if ((limits.size() == 0) || (limits.size() != flags.size()) || (limits.size() != current.size())) {
        throw std::invalid_argument("Limits, flags and current coordinates must be of equal, non-zero size.");
    }

    for (int i = 0; i < limits.size(); i++) {
        if (limits.at(i) == 0) {
            throw std::invalid_argument("All limits should be greater than 0.");
        } else if (current.at(i) >= limits.at(i)) {
            throw std::out_of_range("All coordinates must be smaller than their limits.");
        }
    }

    this->limits = limits;
    this->flags = flags;
    this->current = current;
}

const dim_tuple & Coordinates::getLimits() const { return limits; }
const dim_flags & Coordinates::getFlags() const { return flags; }
const dim_tuple & Coordinates::getCurrent() const { return current; }
void Coordinates::setCurrent(const dim_tuple & value) {
    if (this->limits.size() != value.size()) {
        throw std::invalid_argument("Limits current coordinates must be of equal size.");
    }

    for (int i = 0; i < value.size(); i++) {
        if (value.at(i) >= this->limits.at(i)) {
            throw std::out_of_range("All coordinates must be smaller than their limits.");
        }
    }

    this->current = value;
}

void Coordinates::next() {
    for (std::size_t i = this->limits.size() - 1; i >= 0; i--) {
        if (this->flags.at(i) == DimFlag::fixed) {
            continue;
        }

        this->current.at(i)++;
        if (this->current.at(i) == this->limits.at(i)) {
            this->current.at(i) = 0;
            continue;
        } else {
            return;
        }
    }

    throw std::out_of_range("Already reached maximum possible coordinate.");
}

void Coordinates::previous() {
    for (std::size_t i = this->limits.size() - 1; i >= 0; i--) {
        if (this->flags.at(i) == DimFlag::fixed) {
            continue;
        }

        this->current.at(i)--;
        if (this->current.at(i) == -1) {
            this->current.at(i) = this->limits.at(i) - 1;
            continue;
        } else {
            return;
        }
    }

    throw std::out_of_range("Already reached minimum possible coordinate.");
}

void Coordinates::reset() {
    for (std::size_t i = 0; i < this->limits.size(); i++) {
        if (this->flags.at(i) == DimFlag::fixed) {
            continue;
        }

        this->current.at(i) = 0;
    }
}

bool Coordinates::canIncrease() {
    for (std::size_t i = 0; i < this->limits.size(); i++) {
        if (this->flags.at(i) == DimFlag::fixed) {
            continue;
        }

        if (this->current.at(i) < this->limits.at(i) - 1) {
            return true;
        }
    }

    return false;
}

bool Coordinates::canDecrease() {
    for (std::size_t i = 0; i < this->limits.size(); i++) {
        if (this->flags.at(i) == DimFlag::fixed) {
            continue;
        }

        if (this->current.at(i) > 0) {
            return true;
        }
    }

    return false;
}

Coordinates Coordinates::complement() {
    dim_flags new_flags(this->flags.size());
    for (std::size_t i = 0; i < this->flags.size(); i++) {
        if (this->flags.at(i) == DimFlag::variable) {
            new_flags.at(i) = DimFlag::fixed;
        } else {
            new_flags.at(i) = DimFlag::variable;
        }
    }

    Coordinates new_coordinates(this->limits, new_flags, this->current);
    return new_coordinates;
}


Tensor::Tensor() {
    std::vector<complex> data = { 0.0 };
    std::vector<int> dims = { 1 };
    
    this->id = ++(this->counter);
    this->data = data;
    this->dims = dims;
}

Tensor::Tensor(std::vector<int> dims) {
    int tot_len = 1;
    for (auto d : dims) {
        tot_len *= d;
    }

    std::vector<complex> data(tot_len, 0.0 );

    this->id = ++(this->counter);
    this->data = data;
    this->dims = dims;
}

Tensor::Tensor(std::vector<complex> data, std::vector<int> dims) {
    this->id = ++(this->counter);

    this->data = data;
    this->dims = dims;
}

unsigned int Tensor::counter = 0;

int Tensor::getID() {
    return this->id;
}

std::vector<int> Tensor::getDims() {
    return this->dims;
}

int Tensor::getSize() {
    int size = 1;
    for (auto d : this->getDims()) {
        size *= d;
    }

    return size;
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

void Tensor::setEl(std::vector<int> coords, complex value) {
    std::vector<int> dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    this->data.at(idx) = value;

    return;
}


Bond::Bond(std::pair<Tensor, Tensor> tensors, std::pair<int, int> dims) {
    this->id = ++(this->counter);

    this->tensors = tensors;
    this->dims = dims;
}

unsigned int Bond::counter = 0;

int Bond::getID() {
    return this->id;
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

Tensor TensorNetwork::getTensor(int n) {
    return this->tensors.at(n);
}


std::vector<int> get_contracted_indices(std::vector<int> ids1, std::vector<int> ids2, std::pair<int, int> target_ids) {
    std::vector<int> new_ids(ids1.size() + ids2.size() - 2, 0);
    for (int i = 0, f = 0; i < (int)ids1.size(); i++) {
        if (i != target_ids.first) {
            new_ids.at(i - f) = ids1.at(i - f);
        } else {
            f++;
        }
    }
    for (int i = 0, f = 0; i < (int)ids2.size(); i++) {
        if (i != target_ids.second) {
            new_ids.at(ids1.size() - 1 + i - f) = ids2.at(i - f);
        } else {
            f++;
        }
    }

    return new_ids;
}

int increment_indices(std::vector<int> &ids1, std::vector<int> &ids2, std::vector<int> dims1, std::vector<int> dims2, std::pair<int, int> target_ids) {
    for (int i = (int)ids2.size() - 1; i >= 0; i--) {
        if (i == target_ids.second) {
            continue;
        }

        ids2.at(i)++;
        if (ids2.at(i) == dims2.at(i)) {
            ids2.at(i) = 0;
            continue;
        } else {
            return 1;
        }
    }
    for (int i = (int)ids1.size() - 1; i >= 0; i--) {
        if (i == target_ids.first) {
            continue;
        }

        ids1.at(i)++;
        if (ids1.at(i) == dims1.at(i)) {
            ids1.at(i) = 0;
            continue;
        } else {
            return 1;
        }
    }

    return 0;
}

void TensorNetwork::contract() {
    if (this->bonds.size() == 0) {
        return;
    }

    Bond target_bond = this->bonds.back();
    this->bonds.pop_back();

    std::pair<Tensor, Tensor> target_tensors = target_bond.getTensors();
    std::pair<int, int> target_dims = target_bond.getDims();

    std::vector<int> dims1 = target_tensors.first.getDims();
    std::vector<int> dims2 = target_tensors.second.getDims();

    std::vector<int> new_dims = get_contracted_indices(dims1, dims2, target_dims);

    Tensor new_tensor(new_dims);

    std::vector<int> coords1(dims1.size(), 0);
    std::vector<int> coords2(dims2.size(), 0);
    std::vector<int> new_coords(new_dims.size(), 0);

    int len = dims1.at(target_dims.first);

    int can_increment = 1;
    while (can_increment) {
        complex new_value = 0;
        for (int i = 0; i < len; i++) {
            coords1.at(target_dims.first) = i;
            coords2.at(target_dims.second) = i;
            new_value += target_tensors.first.getEl(coords1) * target_tensors.second.getEl(coords2);
        }

        #ifdef DEBUG
        std::cout << "Calculated element: (";
        for (int i = 0; i < (int)new_coords.size(); i++) {
            std::cout << new_coords.at(i);
            if (i < (int)new_coords.size() - 1) {
                std::cout << ", ";
            }
        }
        std::cout << ") = " << new_value << std::endl;
        #endif

        new_tensor.setEl(new_coords, new_value);

        can_increment = increment_indices(coords1, coords2, dims1, dims2, target_dims);
        new_coords = get_contracted_indices(coords1, coords2, target_dims);
    }
    
    for (int i = 0; i < (int)this->bonds.size(); i++) {
        Bond bond = this->bonds.at(i);
        std::pair<Tensor, Tensor> bond_tensors = bond.getTensors();
        std::pair<int, int> bond_targets = bond.getDims();

        if (bond_tensors.first.getID() == target_tensors.first.getID()) {
            bond_tensors.first = new_tensor;
            bond_targets.first = bond_targets.first - (bond_targets.first > target_dims.first);
        } else if (bond_tensors.first.getID() == target_tensors.second.getID()) {
            bond_tensors.first = new_tensor;
            bond_targets.first = dims1.size() - 1 + bond_targets.first - (bond_targets.first > target_dims.second);
        }

        if (bond_tensors.second.getID() == target_tensors.first.getID()) {
            bond_tensors.second = new_tensor;
            bond_targets.second = bond_targets.second - (bond_targets.second > target_dims.first);
        } else if (bond_tensors.second.getID() == target_tensors.second.getID()) {
            bond_tensors.second = new_tensor;
            bond_targets.second = dims1.size() - 1 + bond_targets.second - (bond_targets.second > target_dims.second);
        }

        this->bonds.at(i) = bond;
    }

    for (int i = 0; i < (int)this->tensors.size(); i++) {
        Tensor tensor = this->tensors.at(i);
        if (tensor.getID() == target_tensors.first.getID() || tensor.getID() == target_tensors.second.getID()) {
            this->tensors.erase(this->tensors.begin() + i);
            i--;
        }
    }

    this->tensors.push_back(new_tensor);

    this->contract();
}

