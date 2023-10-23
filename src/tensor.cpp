#include <cassert>
#include <iostream>
#include <stdexcept>

#include "tensor.hpp"

tidx_tuple def_dims = { 1 };
tels_array def_data(tidx_tuple dims) {
    int tot_len = 1;
    for (auto d : dims) {
        tot_len *= d;
    }
    tels_array data(tot_len, 0.0);
    return data;
}

Tensor::Tensor() : Tensor(def_dims, def_data(def_dims)) {}

Tensor::Tensor(tidx_tuple dims) : Tensor(dims, def_data(dims)) {}

Tensor::Tensor(tidx_tuple dims, tels_array data) : dims(dims), data(data) {
    this->id = ++(this->counter);
}

unsigned int Tensor::counter = 0;

const unsigned int& Tensor::getID() const { return id; }
const tidx_tuple& Tensor::getDims() const { return dims; }

const complex& Tensor::getEl(const tidx_tuple& coords) const {
    tidx_tuple dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    return this->data.at(idx);
}

void Tensor::setEl(const tidx_tuple& coords, const complex& value) {
    tidx_tuple dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    this->data.at(idx) = value;

    return;
}

complex& Tensor::operator[](tidx_tuple coords) {
    tidx_tuple dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    return this->data.at(idx);
}

const complex& Tensor::operator[](tidx_tuple coords) const {
    tidx_tuple dims = this->getDims();

    int idx = 0;
    int base = 1;
    for (int i = coords.size() - 1; i >= 0; --i) {
        idx += coords.at(i) * base;
        base *= dims.at(i);
    }

    return this->data.at(idx);
}

bool Tensor::operator==(const Tensor& rhs) const {
    if (this->dims != rhs.getDims()) {
        return false;
    }

    TIndexing ti(this->dims);
    for (auto i : ti) {
        if ((*this)[i] != rhs[i]) {
            return false;
        }
    }

    return true;
}

bool Tensor::operator!=(const Tensor& rhs) const {
    return !(*this == rhs);
}

std::size_t Tensor::size() {
    std::size_t size = 1;
    for (auto d : this->getDims()) {
        size *= d;
    }

    return size;
}

std::vector<Tensor> Tensor::split(std::size_t along_dim) {
    TIndexing ti(this->dims, along_dim);
    TIndexing ti_res = ti.cut(TIFlag::closed);

    std::vector<Tensor> result;
    for (std::size_t i = 0; i < this->dims.at(along_dim); i++) {
        Tensor t(ti_res.getDims());
        result.push_back(t);
    }

    auto it_res = ti_res.begin();
    for (auto i : ti) {
        for (std::size_t j = 0; j < this->dims.at(along_dim); j++) {
            result.at(j)[*it_res] = (*this)[i];

            if (ti.isLast(i, TIFlag::closed)) {
                break;
            }

            ti.next(i, TIFlag::closed);
        }

        ++it_res;
    }

    return result;
}

std::ostream& operator<<(std::ostream& out, const Tensor& o) {
    tidx_tuple dims = o.getDims();
    TIndexing ti(dims);
    for (auto i : ti) {
        for (std::size_t j = dims.size(); j > 0; j--) {
            if (i.at(j - 1) == 0) {
                out << "(";
            } else {
                break;
            }
        }

        out << o[i];

        bool last_element = true;
        for (std::size_t j = dims.size(); j > 0; j--) {
            if (i.at(j - 1) == dims.at(j - 1) - 1) {
                out << ")";
            } else {
                last_element = false;
                break;
            }
        }

        if (!last_element) {
            out << ", ";
        }
    }

    return out;
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
    tidx_tuple dim1 = t1.getDims();
    tidx_tuple dim2 = t2.getDims();

    TIndexing ti1(dim1, id1);
    TIndexing ti2(dim2, id2);
    TIndexing ti_res = TIndexing::app(ti1.cut(TIFlag::closed), ti2.cut(TIFlag::closed));

    Tensor tres(ti_res.getDims());

    auto it_res = ti_res.begin();
    for (auto i1 : ti1) {
        for (auto i2 : ti2) {
            tres[*it_res] = 0;
            while (true) {
                tres[*it_res] += t1[i1] * t2[i2];

                if (ti1.isLast(i1, TIFlag::closed) && ti1.isLast(i1, TIFlag::closed)) {
                    break;
                }
                
                ti1.next(i1, TIFlag::closed);
                ti2.next(i2, TIFlag::closed);
            }
        
            #ifdef DEBUG
            std::cout << "tres[" << *it_res << "] = " << tres[*it_res] << std::endl;
            #endif
            
            ti1.reset(i1, TIFlag::closed);
            ++it_res;
        }
    }

    return tres;
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

    tidx_tuple dims1 = target_tensors.first.getDims();
    tidx_tuple dims2 = target_tensors.second.getDims();
    
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

