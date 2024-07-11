#include "tensor-new/diag.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated)
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims), truncated_(truncated) {}
  
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated, BcParams params)
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, params), truncated_(truncated) {}

  DiagTensor* DiagTensorBase::toDiag() {
    std::vector<qtnh::tel> els;
    els.reserve(utils::dims_to_size(locOutDims()));

    // ! This is broken for now, as it is impossible to address partial local dimensions. 
    TIndexing ti(locOutDims());
    for (auto idxs : ti) {
      
      els.push_back((*this)[idxs]);
    }

    return new DiagTensor(bc_.env, loc_dims_, dis_dims_, n_dis_in_dims_, truncated_, std::move(els), { bc_.str, bc_.cyc, bc_.off });
  }

  DiagTensor::DiagTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated, std::vector<qtnh::tel>&& diag_els)
    : DiagTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, truncated), loc_diag_els(std::move(diag_els)) {}
  
  DiagTensor::DiagTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated, std::vector<qtnh::tel>&& diag_els, BcParams params)
    : DiagTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, truncated, params), loc_diag_els(std::move(diag_els)) {}
}