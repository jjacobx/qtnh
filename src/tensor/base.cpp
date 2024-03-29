#include <iomanip>

#include "core/utils.hpp"
#include "tensor/base.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  Tensor* Tensor::contract_disp(Tensor*, const std::vector<qtnh::wire>&) {
    utils::throw_unimplemented(); 
    return nullptr; 
  }

  Tensor* Tensor::contract(ConvertTensor*, const std::vector<qtnh::wire>&) {
    utils::throw_unimplemented(); 
    return nullptr; 
  }

  Tensor* Tensor::contract(SDenseTensor*, const std::vector<qtnh::wire>&) {
    utils::throw_unimplemented(); 
    return nullptr; 
  }
  
  Tensor* Tensor::contract(DDenseTensor*, const std::vector<qtnh::wire>&) {
    utils::throw_unimplemented(); 
    return nullptr; 
  }

  Tensor::Tensor(const QTNHEnv& env) 
    : Tensor(env, qtnh::tidx_tup(), qtnh::tidx_tup()) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dist_dims)
    : id(++counter), env(env), active(true), dims(utils::concat_dims(dist_dims, loc_dims)), 
      loc_dims(loc_dims), dist_dims(dist_dims) {};

  std::size_t Tensor::getSize() const { 
    return utils::dims_to_size(getDims()); 
  }

  std::size_t Tensor::getLocSize() const { 
    return utils::dims_to_size(getLocDims()); 
  }

  std::size_t Tensor::getDistSize() const { 
    return utils::dims_to_size(getDistDims()); 
  }

  Tensor* Tensor::contract(Tensor* t1, Tensor* t2, const std::vector<qtnh::wire>& wires) { 
    return t2->contract_disp(t1, wires); 
  }


  namespace ops {
    std::ostream& operator<<(std::ostream& out, const Tensor& o) {
      if (!o.isActive()) {
        out << "Inactive";
        return out;
      }

      out << std::setprecision(2);

      TIndexing ti(o.getLocDims());
      for (auto idxs : ti) {
        out << o.getLocEl(idxs).value();
        if (utils::idxs_to_i(idxs, o.getLocDims()) < o.getLocSize() - 1) {
          out << ", ";
        }
      }

      return out;
    }
  }


  SharedTensor::SharedTensor(qtnh::tidx_tup loc_dims) {
    this->dims = loc_dims;
    this->dist_dims = qtnh::tidx_tup();
    this->loc_dims = loc_dims;
  }

  std::optional<qtnh::tel> SharedTensor::getEl(const qtnh::tidx_tup& idxs) const {
    return getLocEl(idxs); 
  }

  std::optional<qtnh::tel> SharedTensor::getLocEl(const qtnh::tidx_tup& idxs) const {
    return (*this)[idxs]; 
  }
}