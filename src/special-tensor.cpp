#include "core/utils.hpp"
#include "tensor/special.hpp"

namespace qtnh {
  Tensor* SwapTensor::contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) {
    t->swap(wires.at(0).first, wires.at(0).second); 
    return t; 
  }

  Tensor* SwapTensor::contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    t->swap(wires.at(0).first, wires.at(0).second); 
    return t; 
  }

  Tensor* SwapTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) {
    t->swap(wires.at(0).first, wires.at(0).second); 
    return t; 
  }

  SwapTensor::SwapTensor(const QTNHEnv& env, std::size_t n1, std::size_t n2)
    : Tensor(env, { n1, n2, n2, n1 }) {}
  
  qtnh::tel SwapTensor::operator[](const qtnh::tidx_tup& idxs) const {
    return (idxs.at(0) == idxs.at(3)) && (idxs.at(1) == idxs.at(2)); 
  }

  void SwapTensor::swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) {
      utils::throw_unimplemented(); 
      return;
  }


  Tensor* IdentityTensor::contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) { 
    return t; 
  }

  Tensor* IdentityTensor::contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) { 
    return t; 
  }

  Tensor* IdentityTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) { 
    return t; 
  }

  IdentityTensor::IdentityTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims)
    : Tensor(env, utils::concat_dims(in_dims, in_dims)) {};

  qtnh::tel IdentityTensor::operator[](const qtnh::tidx_tup& idxs) const {
    for (std::size_t i = 0; i < idxs.size() / 2; ++i) { 
      if (idxs.at(i) != idxs.at(i + idxs.size() / 2)) { 
        return 0;
      }
    }

    return 1;
  }

  void IdentityTensor::swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) {
    utils::throw_unimplemented(); 
    return; 
  }


  Tensor* DistributeTensor::contract_disp(Tensor* t, const std::vector<qtnh::wire>& wires) { 
    return t; 
  }

  Tensor* DistributeTensor::contract(SDenseTensor* t, const std::vector<qtnh::wire>& wires) { 
    return t; 
  }

  Tensor* DistributeTensor::contract(DDenseTensor* t, const std::vector<qtnh::wire>& wires) { 
    return t; 
  }

  DistributeTensor::DistributeTensor(const QTNHEnv& env, const qtnh::tidx_tup& in_dims)
    : Tensor(env, utils::concat_dims(in_dims, in_dims)) {};
  
  std::optional<qtnh::tel> DistributeTensor::getEl(const qtnh::tidx_tup& idxs) const {
    return getLocEl(idxs);
  }

  std::optional<qtnh::tel> DistributeTensor::getLocEl(const qtnh::tidx_tup& idxs) const {
    return (*this)[idxs];
  }

  qtnh::tel DistributeTensor::operator[](const qtnh::tidx_tup& idxs) const {
    for (std::size_t i = 0; i < idxs.size() / 2; ++i) { 
      if (idxs.at(i) != idxs.at(i + idxs.size() / 2)) { 
        return 0;
      }
    }

    return 1;
  }

  void DistributeTensor::swap(qtnh::tidx_tup_st, qtnh::tidx_tup_st) {
    utils::throw_unimplemented(); 
    return; 
  }
}