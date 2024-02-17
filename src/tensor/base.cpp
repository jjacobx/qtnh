#include "core/utils.hpp"
#include "tensor/base.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  Tensor* Tensor::contract_disp(Tensor*, const std::vector<qtnh::wire>&) {
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

  Tensor::Tensor(const QTNHEnv& env, const qtnh::tidx_tup& dims)
    : id(++counter), env(env), active(true), dims(dims), loc_dims(dims), dist_dims(qtnh::tidx_tup()) {};

  std::size_t Tensor::getSize() const { 
    return utils::dims_to_size(getDims()); 
  }

  std::size_t Tensor::getLocSize() const { 
    return utils::dims_to_size(getLocDims()); 
  }

  std::size_t Tensor::getDistSize() const { 
    return utils::dims_to_size(getDistDims()); 
  }

  std::optional<qtnh::tel> Tensor::getEl(const qtnh::tidx_tup& idxs) const {
    return getLocEl(idxs); 
  }

  std::optional<qtnh::tel> Tensor::getLocEl(const qtnh::tidx_tup& idxs) const {
    return (*this)[idxs]; 
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
}