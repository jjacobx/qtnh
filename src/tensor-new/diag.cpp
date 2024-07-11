#include "tensor-new/diag.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated)
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims), truncated_(truncated) {}
  
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated, BcParams params)
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, params), truncated_(truncated) {}

  DiagTensor* DiagTensorBase::toDiag() {

  }
}