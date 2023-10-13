#include <cassert>
#include <iostream>
#include <stdexcept>

#include "tensor.hpp"

dim_flags def_flags(std::size_t n) {
    dim_flags flags(n, DimFlag::variable);
    return flags;
}

dim_tuple def_current(std::size_t n) {
    dim_tuple current(n, 0);
    return current;
}

Coordinates::Coordinates(dim_tuple limits) : Coordinates(limits, def_flags(limits.size()), def_current(limits.size())) {}

Coordinates::Coordinates(dim_tuple limits, std::size_t constant_dim) : Coordinates(limits) {
    if (constant_dim >= limits.size()) {
        throw std::out_of_range("Constant dimension index must fit within limits.");
    }

    this->flags.at(constant_dim) = DimFlag::fixed;
}

Coordinates::Coordinates(dim_tuple limits, dim_flags flags) : Coordinates(limits, flags, def_current(limits.size())) {}

Coordinates::Coordinates(dim_tuple limits, dim_flags flags, dim_tuple current) : limits(limits), flags(flags), current(current) {
    if ((limits.size() == 0) || (limits.size() != flags.size()) || (limits.size() != current.size())) {
        throw std::invalid_argument("Limits, flags and current coordinates must be of equal, non-zero size.");
    }

    for (std::size_t i = 0; i < limits.size(); i++) {
        if (limits.at(i) == 0) {
            throw std::invalid_argument("All limits should be greater than 0.");
        } else if (current.at(i) >= limits.at(i)) {
            throw std::out_of_range("All coordinates must be smaller than their limits.");
        }
    }
}

const dim_tuple& Coordinates::getLimits() const { return limits; }
const dim_flags& Coordinates::getFlags() const { return flags; }
const dim_tuple& Coordinates::getCurrent() const { return current; }
void Coordinates::setCurrent(const dim_tuple & value) {
    if (this->limits.size() != value.size()) {
        throw std::invalid_argument("Limits and current coordinates must be of equal size.");
    }

    for (std::size_t i = 0; i < value.size(); i++) {
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

        if (this->current.at(i) == this->limits.at(i) - 1) {
            this->current.at(i) = 0;
            continue;
        } else {
            this->current.at(i)++;
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

        if (this->current.at(i) == 0) {
            this->current.at(i) = this->limits.at(i) - 1;
            continue;
        } else {
            this->current.at(i)--;
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

Coordinates Coordinates::dropFixed() {
    dim_tuple new_limits = this->limits;
    dim_flags new_flags = this->flags;
    dim_tuple new_current = this->current;

    int rm_count = 0;
    for(std::size_t i = 0; i < this->flags.size(); i++) {
        if (this->flags.at(i) == DimFlag::fixed) {
            new_limits.erase(new_limits.begin() + i - rm_count);
            new_flags.erase(new_flags.begin() + i - rm_count);
            new_current.erase(new_current.begin() + i - rm_count);
            rm_count++;
        }
    }

    return Coordinates(new_limits, new_flags, new_current);
}

Coordinates Coordinates::operator&&(const Coordinates& rhs) const {
    dim_tuple new_limits = this->limits;
    dim_flags new_flags = this->flags;
    dim_tuple new_current = this->current;

    new_limits.insert(new_limits.end(), rhs.getLimits().begin(), rhs.getLimits().end());
    new_flags.insert(new_flags.end(), rhs.getFlags().begin(), rhs.getFlags().end());
    new_current.insert(new_current.end(), rhs.getCurrent().begin(), rhs.getCurrent().end());

    return Coordinates(new_limits, new_flags, new_current);
}


std::ostream& operator<<(std::ostream& out, const Coordinates& o) {
    dim_tuple current = o.getCurrent();

    out << "(";
    for (std::size_t i = 0; i < current.size(); i++) {
        out << current.at(i);
        if (i < current.size() - 1) {
            out << ", ";
        }
    }
    out << ")";

    return out;
}


dim_tuple def_dims = { 1 };
std::vector<complex> def_data(dim_tuple dims) {
    int tot_len = 1;
    for (auto d : dims) {
        tot_len *= d;
    }
    std::vector<complex> data(tot_len, 0.0);
    return data;
}

Tensor::Tensor() : Tensor(def_dims, def_data(def_dims)) {}

Tensor::Tensor(dim_tuple dims) : Tensor(dims, def_data(dims)) {}

Tensor::Tensor(dim_tuple dims, std::vector<complex> data) : dims(dims), data(data) {
    this->id = ++(this->counter);
}

unsigned int Tensor::counter = 0;

const unsigned int& Tensor::getID() const { return id; }
const dim_tuple& Tensor::getDims() const { return dims; }

const complex& Tensor::getEl(const dim_tuple& coords) const {
    dim_tuple dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    return this->data.at(idx);
}

void Tensor::setEl(const dim_tuple& coords, const complex& value) {
    dim_tuple dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    this->data.at(idx) = value;

    return;
}

std::size_t Tensor::size() {
    std::size_t size = 1;
    for (auto d : this->getDims()) {
        size *= d;
    }

    return size;
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

Tensor contract2(Tensor t1, Tensor t2, int id1, int id2) {
    dim_tuple dim1 = t1.getDims();
    dim_tuple dim2 = t2.getDims();

    Coordinates c1(dim1, id1);
    Coordinates c2(dim2, id2);
    Coordinates c_new = c1.dropFixed() && c2.dropFixed();

    Tensor t_new(c_new.getLimits());

    while (true) {
        Coordinates c1p = c1.complement();
        Coordinates c2p = c2.complement();
        complex el_new = 0.0;

        while (true) {
            el_new += t1.getEl(c1p.getCurrent()) * t2.getEl(c2p.getCurrent());
            if (c1p.canIncrease() && c2p.canIncrease()) {
                c1p.next();
                c2p.next();
            } else {
                break;
            }
        }

        #ifdef DEBUG
        std::cout << c_new << " = " << el_new << std::endl;
        #endif

        t_new.setEl(c_new.getCurrent(), el_new);
        if (c2.canIncrease()) {
            c2.next();
            c_new.next();
        } else if (c1.canIncrease()) {
            c1.next();
            c2.reset();
            c_new.next();
        } else {
            break;
        }
        
    }

    return t_new;
} 

void TensorNetwork::contract() {
    if (this->bonds.size() == 0) {
        return;
    }

    Bond target_bond = this->bonds.back();
    this->bonds.pop_back();

    std::pair<Tensor, Tensor> target_tensors = target_bond.getTensors();
    std::pair<int, int> target_dims = target_bond.getDims();

    Tensor new_tensor = contract2(target_tensors.first, target_tensors.second, target_dims.first, target_dims.second);

    dim_tuple dims1 = target_tensors.first.getDims();
    dim_tuple dims2 = target_tensors.second.getDims();
    
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

