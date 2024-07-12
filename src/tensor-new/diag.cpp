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
    : DiagTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, truncated), loc_diag_els_(std::move(diag_els)) {}
  
  DiagTensor::DiagTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated, std::vector<qtnh::tel>&& diag_els, BcParams params)
    : DiagTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, truncated, params), loc_diag_els_(std::move(diag_els)) {}


  IdenTensor::IdenTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated)
    : DiagTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, truncated) {}

  IdenTensor::IdenTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, bool truncated, BcParams params)
    : DiagTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, truncated, params) {}

  qtnh::tel IdenTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto dis_idxs = utils::i_to_idxs(bc_.group_id, dis_dims_);
    auto tot_idxs = utils::concat_dims(dis_idxs, loc_idxs);

    auto [idxs1, idxs2] = utils::split_dims(tot_idxs, tot_idxs.size() / 2);

    for (int i = 0; i < idxs1.size(); ++i) {
      if (idxs1.at(i) != idxs2.at(i)) return 0;
    }

    return 1;
  }

  qtnh::tel IdenTensor::at(qtnh::tidx_tup tot_idxs) const {
    auto [tot_in_idxs, tot_out_idxs] = utils::split_dims(tot_idxs, tot_idxs.size() / 2);

    for (int i = 0; i < tot_in_idxs.size(); ++i) {
      if (tot_in_idxs.at(i) != tot_out_idxs.at(i)) return 0;
    }

    return 1;
  }

  IdenTensor* IdenTensor::rebcast(BcParams params) {
    Broadcaster new_bc(bc_.env, bc_.base, params);
    bc_ = std::move(new_bc);

    return this;
  }
}