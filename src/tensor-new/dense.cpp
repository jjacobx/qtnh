#include "tensor-new/dense.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  DenseTensorBase::DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims)
    : Tensor(env, loc_dims, dis_dims) {}

  DenseTensorBase::DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, DistParams params)
    : Tensor(env, loc_dims, dis_dims, params) {}

  DenseTensor* DenseTensorBase::toDense() {
    std::vector<qtnh::tel> els;
    els.reserve(locSize());

    TIndexing ti(locDims());
    for (auto idxs : ti) {
      els.push_back((*this)[idxs]);
    }

    // ? Is it better to use local members or accessors? 
    return new DenseTensor(dist_.env, loc_dims_, dis_dims_, els, DistParams { dist_.stretch, dist_.cycles, dist_.offset });
  }

  DenseTensor::DenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel> els)
    : DenseTensorBase(env, loc_dims, dis_dims), loc_els(els) {}

  DenseTensor::DenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel> els, DistParams params)
    : DenseTensorBase(env, loc_dims, dis_dims, params), loc_els(els) {}

  qtnh::tel DenseTensor::operator[](const qtnh::tidx_tup& loc_idxs) const {
    // TODO
  }

  qtnh::tel& DenseTensor::operator[](const qtnh::tidx_tup& loc_idxs) {
    // TODO
  }

  void DenseTensor::put(const qtnh::tidx_tup& loc_idxs, qtnh::tel el) {
    // TODO
  }

  Tensor* DenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    // TODO
  }

  Tensor* DenseTensor::redistribute(DistParams params) {
    // TODO
  }

  Tensor* DenseTensor::repile(std::vector<qtnh::tidx_tup_st> idx_locs) {
    // TODO
  }
}