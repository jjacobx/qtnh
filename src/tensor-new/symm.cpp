#include "tensor-new/symm.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  SymmTensorBase::SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims) 
    : DenseTensorBase(env, loc_dims, dis_dims), n_dis_in_dims_(n_dis_in_dims) {}

  SymmTensorBase::SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, BcParams params) 
    : DenseTensorBase(env, loc_dims, dis_dims, params), n_dis_in_dims_(n_dis_in_dims) {}



  SymmTensor::SymmTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel>&& els) 
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims), loc_els_(std::move(els)) {}

  SymmTensor::SymmTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel>&& els, BcParams params) 
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, params), loc_els_(std::move(els)) {}
}