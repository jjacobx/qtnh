#include "tensor/diag.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool truncated)
    : SymmTensorBase(env, dis_dims, loc_dims), truncated_(truncated) {}
  
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool truncated, BcParams params)
    : SymmTensorBase(env, dis_dims, loc_dims, params), truncated_(truncated) {}

  // Specialised convert template from tensor header requires full class definition. 
  template<> 
  std::unique_ptr<DiagTensor> Tensor::convert<DiagTensor>(tptr tp) {
    return utils::one_unique(std::move(tp), tp->toDiag()); 
  }

  DiagTensor* DiagTensorBase::toDiag() noexcept {
    std::vector<qtnh::tel> els;
    els.reserve(utils::dims_to_size(utils::halve_dims(locDims())));

    std::vector<TIFlag> ifls(totDims().size());
    for (std::size_t i = 0; i < totDims().size(); ++i) {
      if (i < disDims().size() / 2) {
        ifls.at(i) = { "distributed-in", 0 };
      } else if (i < disDims().size()) {
        ifls.at(i) = { "distributed-out", 0 };
      } else if (i < disDims().size() + locDims().size() / 2) {
        ifls.at(i) = { "local-in", 0 };
      } else {
        ifls.at(i) = { "local-out", 0 };
      }
    }

    auto curr_dis_idxs = utils::i_to_idxs(bc_.group_id, dis_dims_);

    // ! This is broken for now, as diagonal tensors are not yet implemented. 
    TIndexing ti(utils::halve_dims(locDims()));
    for (auto idxs : ti.tup("local-in")) {
      idxs = ti.next(idxs, "local-out");
      els.push_back(this->at(idxs));
    }

    return new DiagTensor(bc_.env, dis_dims_, loc_dims_, truncated_, std::move(els), { bc_.str, bc_.cyc, bc_.off });
  }


  Tensor* DiagTensorBase::swapIO(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    return toDiag()->swap(idx1, idx2);
  }

  Tensor* DiagTensorBase::rebcast(BcParams params) {
    return toDiag()->rebcast(params);
  }

  Tensor* DiagTensorBase::rescatterIO(int offset) {
    return toDiag()->rescatter(offset);
  }

  DiagTensor::DiagTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool truncated, std::vector<qtnh::tel>&& diag_els)
    : DiagTensorBase(env, dis_dims, loc_dims, truncated), diagonal_(env, utils::halve_dims(dis_dims), utils::halve_dims(loc_dims), std::move(diag_els)) {}
  
  DiagTensor::DiagTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool truncated, std::vector<qtnh::tel>&& diag_els, BcParams params)
    : DiagTensorBase(env, dis_dims, loc_dims, truncated, params), diagonal_(env, utils::halve_dims(dis_dims), utils::halve_dims(loc_dims), std::move(diag_els)) {}

  std::unique_ptr<Tensor> DiagTensor::copy() const noexcept {
    auto els = diagonal_.loc_els_;
    auto tp = new DiagTensor(bc_.env, dis_dims_, loc_dims_, truncated_, std::move(els));
    return std::unique_ptr<DiagTensor>(tp);
  }

  qtnh::tel DiagTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto dis_idxs = utils::i_to_idxs(bc_.group_id, dis_dims_);
    auto [dis_idxs_in, dis_idxs_out] = utils::split_dims(dis_idxs, dis_idxs.size() / 2);
    auto [loc_idxs_in, loc_idxs_out] = utils::split_dims(loc_idxs, loc_idxs.size() / 2);

    if ((dis_idxs_in != dis_idxs_out) || (loc_idxs_in != loc_idxs_out)) {
      return 0;
    } else {
      return diagonal_.at(loc_idxs_out);
    }
  }

  qtnh::tel DiagTensor::at(qtnh::tidx_tup tot_idxs) const {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    auto [dis_idxs_in, dis_idxs_out] = utils::split_dims(dis_idxs, dis_idxs.size() / 2);
    auto [loc_idxs_in, loc_idxs_out] = utils::split_dims(loc_idxs, loc_idxs.size() / 2);

    auto tot_idxs_in = utils::concat_dims(dis_idxs_in, loc_idxs_in);
    auto tot_idxs_out = utils::concat_dims(dis_idxs_out, loc_idxs_out);

    if (tot_idxs_in != tot_idxs_out) {
      return 0;
    } else {
      return diagonal_.at(tot_idxs_out);
    }
  }

  void DiagTensor::put(qtnh::tidx_tup tot_idxs, qtnh::tel el) {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    auto [dis_idxs_in, dis_idxs_out] = utils::split_dims(dis_idxs, dis_idxs.size() / 2);
    auto [loc_idxs_in, loc_idxs_out] = utils::split_dims(loc_idxs, loc_idxs.size() / 2);

    auto tot_idxs_in = utils::concat_dims(dis_idxs_in, loc_idxs_in);
    auto tot_idxs_out = utils::concat_dims(dis_idxs_out, loc_idxs_out);

    if (tot_idxs_in != tot_idxs_out) {
      throw std::runtime_error("Tried to insert non-diagonal element.");
    } else {
      diagonal_.put(tot_idxs_out, el);
    }
  }

  DiagTensor* DiagTensor::swapIO(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    diagonal_._swap_internal(&diagonal_, idx1, idx2);
    return this;
  }

  DiagTensor* DiagTensor::rebcast(BcParams params) {
    diagonal_._rebcast_internal(&diagonal_, params);
    return this;
  }

  DiagTensor* DiagTensor::rescatterIO(int offset) {
    diagonal_._rescatter_internal(&diagonal_, offset);
    return this;
  }

  IdenTensor::IdenTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool truncated)
    : DiagTensorBase(env, dis_dims, loc_dims, truncated) {}

  IdenTensor::IdenTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, bool truncated, BcParams params)
    : DiagTensorBase(env, dis_dims, loc_dims, truncated, params) {}

  std::unique_ptr<Tensor> IdenTensor::copy() const noexcept {
    auto tp = new IdenTensor(bc_.env, dis_dims_, loc_dims_, truncated_, { bc_.str, bc_.cyc, bc_.off });
    return std::unique_ptr<IdenTensor>(tp);
  }

  qtnh::tel IdenTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto dis_idxs = utils::i_to_idxs(bc_.group_id, dis_dims_);
    auto tot_idxs = utils::concat_dims(dis_idxs, loc_idxs);

    auto [idxs1, idxs2] = utils::split_dims(tot_idxs, tot_idxs.size() / 2);

    for (std::size_t i = 0; i < idxs1.size(); ++i) {
      if (idxs1.at(i) != idxs2.at(i)) return 0;
    }

    return 1;
  }

  qtnh::tel IdenTensor::at(qtnh::tidx_tup tot_idxs) const {
    auto [tot_in_idxs, tot_out_idxs] = utils::split_dims(tot_idxs, tot_idxs.size() / 2);

    for (std::size_t i = 0; i < tot_in_idxs.size(); ++i) {
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