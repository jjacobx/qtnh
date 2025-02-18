#include <iostream>

#include "tensor/diag.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, 
                                 bool shrunk)
  : SymmTensorBase(env, dis_dims, loc_dims)
  , shrunk_(shrunk) 
  {}
  
  DiagTensorBase::DiagTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, 
                                 bool shrunk, BcParams params)
  : SymmTensorBase(env, dis_dims, loc_dims, params)
  , shrunk_(shrunk)
  {}

  // Specialised convert template from tensor header requires full class definition. 
  template<> 
  std::unique_ptr<DiagTensor> Tensor::convert<DiagTensor>(tptr tp) {
    auto p = tp->toDiag();
    return utils::one_unique(std::move(tp), p); 
  }

  bool DiagTensorBase::available(qtnh::tidx_tup tot_idxs) const noexcept {
    if (!has(tot_idxs)) {
      return false;
    } else if (!shrunk_) {
      return true;
    }

    auto dis_idxs = utils::i_to_idxs(bc_.gid(), dis_dims_);
    auto [dis_idxs_in, dis_idxs_out] = utils::split_dims(dis_idxs, dis_idxs.size() / 2);
    auto dis_dims_in = utils::split_dims(dis_dims_, dis_dims_.size() / 2).first;

    return utils::idxs_to_i(dis_idxs_in, dis_dims_in) == 0;
  }

  void DiagTensor::reshape(qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims) {
    auto [dis_in_dims, dis_out_dims] = utils::split_dims(dis_dims, dis_dims.size() / 2);
    auto [loc_in_dims, loc_out_dims] = utils::split_dims(loc_dims, loc_dims.size() / 2);
    if ((dis_in_dims == dis_out_dims) && (loc_in_dims == loc_out_dims)) {
      diagonal_.reshape(dis_out_dims, loc_out_dims);
      Tensor::reshape(dis_dims, loc_dims);
    } else {
      throw std::invalid_argument("Invalid symmetric dimensions.");
    }
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

    auto curr_dis_idxs = utils::i_to_idxs(bc_.gid(), dis_dims_);

    // ! This is broken for now, as diagonal tensors are not yet implemented. 
    TIndexing ti(utils::halve_dims(locDims()));
    for (auto idxs : ti.tup("local-in")) {
      idxs = ti.next(idxs, "local-out");
      els.push_back(this->at(idxs));
    }

    return new DiagTensor(bc_.env(), dis_dims_, loc_dims_, shrunk_, std::move(els), bc_.params());
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

  Tensor* DiagTensorBase::shrink() {
    return toDiag()->shrink();
  }

  Tensor* DiagTensorBase::expand() {
    return toDiag()->expand();
  }

  DiagTensor::DiagTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, 
                         bool shrunk, std::vector<qtnh::tel>&& diag_els)
  : DiagTensorBase(env, dis_dims, loc_dims, shrunk)
  , diagonal_(env, utils::halve_dims(dis_dims), utils::halve_dims(loc_dims), std::move(diag_els)) 
  {}
  
  DiagTensor::DiagTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, 
                         bool shrunk, std::vector<qtnh::tel>&& diag_els, BcParams params)
  : DiagTensorBase(env, dis_dims, loc_dims, shrunk, params)
  , diagonal_(env, utils::halve_dims(dis_dims), utils::halve_dims(loc_dims), std::move(diag_els)) 
  {}

  std::unique_ptr<Tensor> DiagTensor::copy() const noexcept {
    auto els = diagonal_.loc_els_;
    auto tp = new DiagTensor(bc_.env(), dis_dims_, loc_dims_, shrunk_, std::move(els));
    return std::unique_ptr<DiagTensor>(tp);
  }

  qtnh::tel DiagTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto dis_idxs = utils::i_to_idxs(bc_.gid(), dis_dims_);
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

    dis_dims_ = utils::concat_dims(diagonal_.disDims(), diagonal_.disDims());
    loc_dims_ = utils::concat_dims(diagonal_.locDims(), diagonal_.locDims());

    return this;
  }

  DiagTensor* DiagTensor::rebcast(BcParams params) {
    auto diag_params = params;
    if (!shrunk_) diag_params.cyc *= qtnh::uint(diagonal_.disSize());
    diagonal_._rebcast_internal(&diagonal_, diag_params);

    // Update broadcasters
    diagonal_.bc_ = { diagonal_.bc_.env(), diagonal_.bc_.base(), diag_params };
    bc_ = { bc_.env(), bc_.base(), params };

    return this;
  }

  DiagTensor* DiagTensor::rescatterIO(int offset) {
    diagonal_._rescatter_internal(&diagonal_, offset);

    // Update dimensions and broadcaster
    if (offset < 0) {
      auto loc_dims2 = qtnh::tidx_tup(dis_dims_.end() + offset, dis_dims_.end());
      auto shift = (qtnh::uint)utils::dims_to_size(loc_dims2);

      diagonal_.loc_dims_.insert(diagonal_.loc_dims_.begin(), loc_dims2.begin(), loc_dims2.end());
      diagonal_.dis_dims_.erase(diagonal_.dis_dims_.end() + offset, diagonal_.dis_dims_.end());

      auto out_params = diagonal_.bc_.params();
      out_params.str *= shift;
      diagonal_.bc_ = { diagonal_.bc_.env(), (qtnh::uint)diagonal_.disSize(), out_params };
    } else if (offset > 0) {
      auto dis_dims2 = qtnh::tidx_tup(diagonal_.loc_dims_.begin(), diagonal_.loc_dims_.begin() + offset);
      auto shift = utils::dims_to_size(dis_dims2);

      diagonal_.loc_dims_.erase(diagonal_.loc_dims_.begin(), diagonal_.loc_dims_.begin() + offset);
      diagonal_.dis_dims_.insert(diagonal_.dis_dims_.end(), dis_dims2.begin(), dis_dims2.end());

      // Resize base of broadcaster first. Have to re-create the communicator. 
      diagonal_.bc_ = { diagonal_.bc_.env(), (qtnh::uint)diagonal_.disSize(), diagonal_.bc_.params() };

      auto out_params = diagonal_.bc_.params();
      out_params.str = (qtnh::uint)std::max(1UL, out_params.str / shift);
      out_params.cyc *= (qtnh::uint)shift;
      diagonal_._rebcast_internal(&diagonal_, out_params);
      diagonal_.bc_ = { diagonal_.bc_.env(), (qtnh::uint)diagonal_.disSize(), out_params };
    }

    dis_dims_ = utils::concat_dims(diagonal_.disDims(), diagonal_.disDims());
    loc_dims_ = utils::concat_dims(diagonal_.locDims(), diagonal_.locDims());

    auto dis_size = diagonal_.disSize();
    auto params = diagonal_.bc_.params();

    if (!shrunk_) { 
      params.cyc /= qtnh::uint(dis_size);
      dis_size *= dis_size;
    }

    bc_ = { bc_.env(), (qtnh::uint)dis_size, params };
    
    return this;
  }

  DiagTensor* DiagTensor::shrink() {
    if (shrunk_) return this;

    auto diag_params = diagonal_.bc_.params();
    diag_params.cyc /= qtnh::uint(diagonal_.disSize());

    diagonal_._rebcast_internal(&diagonal_, diag_params);
    diagonal_.bc_ = { diagonal_.bc_.env(), diagonal_.bc_.base(), diag_params };

    shrunk_ = true;
    return this;
  }

  DiagTensor* DiagTensor::expand() {
    if (!shrunk_) return this;

    auto diag_params = diagonal_.bc_.params();
    diag_params.cyc *= qtnh::uint(diagonal_.disSize());

    diagonal_._rebcast_internal(&diagonal_, diag_params);
    diagonal_.bc_ = { diagonal_.bc_.env(), diagonal_.bc_.base(), diag_params };

    shrunk_ = false;
    return this;
  }

  IdenTensor::IdenTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, 
                         bool shrunk)
  : DiagTensorBase(env, dis_dims, loc_dims, shrunk)
  {}

  IdenTensor::IdenTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, 
                         bool shrunk, BcParams params)
  : DiagTensorBase(env, dis_dims, loc_dims, shrunk, params)
  {}

  std::unique_ptr<Tensor> IdenTensor::copy() const noexcept {
    auto tp = new IdenTensor(bc_.env(), dis_dims_, loc_dims_, shrunk_, bc_.params());
    return std::unique_ptr<IdenTensor>(tp);
  }

  qtnh::tel IdenTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto dis_idxs = utils::i_to_idxs(bc_.gid(), dis_dims_);
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
    Broadcaster new_bc(bc_.env(), bc_.base(), params);
    bc_ = std::move(new_bc);

    return this;
  }
}
