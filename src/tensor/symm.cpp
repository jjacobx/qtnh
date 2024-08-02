#include "tensor/symm.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  SymmTensorBase::SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, qtnh::tidx_tup_st n_dis_in_dims) 
    : DenseTensorBase(env, dis_dims, loc_dims), n_dis_in_dims_(n_dis_in_dims) {}

  SymmTensorBase::SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, qtnh::tidx_tup_st n_dis_in_dims, BcParams params) 
    : DenseTensorBase(env, dis_dims, loc_dims, params), n_dis_in_dims_(n_dis_in_dims) {}

  // Specialised convert template from tensor header requires full class definition. 
  template<> 
  std::unique_ptr<SymmTensor> Tensor::convert<SymmTensor>(tptr tp) {
    return utils::one_unique(std::move(tp), tp->toSymm()); 
  }

  SymmTensor* SymmTensorBase::toSymm() noexcept {
    std::vector<qtnh::tel> els;
    els.reserve(locSize());

    std::vector<TIFlag> ifls(totDims().size(), { "local", 0 });
    for (std::size_t i = 0; i < dis_dims_.size(); ++i) {
      ifls.at(i) = { "distributed", 0 };
    }

    auto curr_idxs = utils::concat_dims(utils::i_to_idxs(bc_.group_id, dis_dims_), qtnh::tidx_tup(loc_dims_.size(), 0));

    TIndexing ti(totDims(), ifls);
    for (auto idxs : ti.tup("local", curr_idxs)) {
      els.push_back(this->at(idxs));
    }

    return new SymmTensor(bc_.env, dis_dims_, loc_dims_, n_dis_in_dims_, std::move(els), BcParams { bc_.str, bc_.cyc, bc_.off });
  }


  Tensor* SymmTensorBase::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2, TIdxIO io) {
    return toSymm()->swap(idx1, idx2, io);
  }

  Tensor* SymmTensorBase::rebcast(BcParams params) {
    return toSymm()->rebcast(params);
  }

  Tensor* SymmTensorBase::rescatter(int offset, TIdxIO io) {
    return toSymm()->rescatter(offset, io);
  }

  Tensor* SymmTensorBase::permute(std::vector<qtnh::tidx_tup_st> ptup, TIdxIO io) {
    return this->toSymm()->permute(ptup, io);
  }

  SymmTensor::SymmTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel>&& els) 
    : SymmTensorBase(env, dis_dims, loc_dims, n_dis_in_dims), TIDense(std::move(els)) {}

  SymmTensor::SymmTensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel>&& els, BcParams params) 
    : SymmTensorBase(env, dis_dims, loc_dims, n_dis_in_dims, params), TIDense(std::move(els)) {}

  std::unique_ptr<Tensor> SymmTensor::copy() const noexcept {
    auto els = loc_els_;
    auto tp = new SymmTensor(bc_.env, dis_dims_, loc_dims_, n_dis_in_dims_, std::move(els), { bc_.str, bc_.cyc, bc_.off });
    return std::unique_ptr<SymmTensor>(tp);
  }

  qtnh::tel SymmTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto i = utils::idxs_to_i(loc_idxs, loc_dims_);
    return loc_els_.at(i);
  }

  qtnh::tel& SymmTensor::operator[](qtnh::tidx_tup loc_idxs) {
    auto i = utils::idxs_to_i(loc_idxs, loc_dims_);
    return loc_els_.at(i);
  }

  qtnh::tel SymmTensor::at(qtnh::tidx_tup tot_idxs) const {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    if (bc_.group_id != (int)utils::idxs_to_i(dis_idxs, dis_dims_)) {
      throw std::invalid_argument("Element at given indices is not present on calling rank. ");
    }

    return loc_els_.at(utils::idxs_to_i(loc_idxs, loc_dims_));
  }

  qtnh::tel& SymmTensor::at(qtnh::tidx_tup tot_idxs) {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    if (bc_.group_id != (int)utils::idxs_to_i(dis_idxs, dis_dims_)) {
      throw std::invalid_argument("Element at given indices is not present on calling rank. ");
    }

    return loc_els_.at(utils::idxs_to_i(loc_idxs, loc_dims_));
  }

  void SymmTensor::put(qtnh::tidx_tup tot_idxs, qtnh::tel el) {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    auto target_id = (int)utils::idxs_to_i(dis_idxs, dis_dims_);

    int call_id;
    MPI_Comm_rank(bc_.group_comm, &call_id);

    if (call_id == target_id) {
      auto i = utils::idxs_to_i(loc_idxs, loc_dims_);
      loc_els_.at(i) = el;
    }
  }

  SymmTensor* SymmTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2, TIdxIO io) {
    // Convert to general swap indices
    if (io == TIdxIO::in) {
      if (idx1 >= disInDims().size()) { 
        idx1 += disOutDims().size();
      }
      if (idx2 >= disInDims().size()) { 
        idx2 += disOutDims().size();
      }
    } else {
      if (idx1 >= disOutDims().size()) { 
        idx1 += locInDims().size();
      }
      if (idx2 >= disInDims().size()) { 
        idx2 += locInDims().size();
      }
      idx1 += disInDims().size();
      idx2 += disInDims().size();
    }

    _swap_internal(this, idx1, idx2);
    return this;
  }

  SymmTensor* SymmTensor::rebcast(BcParams params) {
    _rebcast_internal(this, params);
    
    // Update broadcaster
    Broadcaster new_bc(bc_.env, bc_.base, params);
    bc_ = std::move(new_bc);

    return this;
  }

  // TODO: Implement using permute
  SymmTensor* SymmTensor::rescatter(int offset, TIdxIO io) {
    if (offset < 0) {
      if (io == TIdxIO::in) {
        _shift_internal(this, disInDims().size() + offset, disDims().size(), offset);
        n_dis_in_dims_ += offset;
      }

      _rescatter_internal(this, offset);

      auto loc_dims2 = qtnh::tidx_tup(dis_dims_.end() + offset, dis_dims_.end());
      auto shift = (qtnh::uint)utils::dims_to_size(loc_dims2);

      loc_dims_.insert(loc_dims_.begin(), loc_dims2.begin(), loc_dims2.end());
      dis_dims_.erase(dis_dims_.end() - offset, dis_dims_.end());

      Broadcaster new_bc(bc_.env, locSize(), { bc_.str * shift, bc_.cyc, bc_.off });
      bc_ = std::move(new_bc);

      if (io == TIdxIO::out) {
        _shift_internal(this, disDims().size(), disDims().size() + locInDims().size() + offset, offset);
      }
    }
    else if (offset > 0) {
      if (io == TIdxIO::out) {
        _shift_internal(this, disDims().size(), disDims().size() + locInDims().size() + offset, offset);
      }

      _rescatter_internal(this, offset);

      auto dis_dims2 = qtnh::tidx_tup(loc_dims_.begin(), loc_dims_.begin() + offset);
      auto shift = utils::dims_to_size(dis_dims2);

      BcParams params(std::max(1UL, bc_.str / shift), bc_.cyc, bc_.off);

      loc_dims_.erase(loc_dims_.begin(), loc_dims_.begin() + offset);
      dis_dims_.insert(dis_dims_.end(), dis_dims2.begin(), dis_dims2.end());

      Broadcaster new_bc(bc_.env, locSize(), params);
      bc_ = std::move(new_bc);

      if (io == TIdxIO::in) {
        _shift_internal(this, disInDims().size(), disDims().size(), offset);
        n_dis_in_dims_ += offset;
      }
    }

    return this;
  }

  SymmTensor* SymmTensor::permute(std::vector<qtnh::tidx_tup_st> ptup, TIdxIO io) {
    std::vector<qtnh::tidx_tup_st> ptup_full(totDims().size());
    for (std::size_t i = 0; i < ptup_full.size(); ++i) {
      ptup_full.at(i) = i;
    }
    
    std::size_t cutoff, offset1, offset2;
    if (io == TIdxIO::in) {
      cutoff = disInDims().size();
      offset1 = 0;
      offset2 = disOutDims().size();
    } else {
      cutoff = disDims().size();
      offset1 = disInDims().size();
      offset2 = locInDims().size();
    }

    for (std::size_t i = 0; i < ptup.size(); ++i) {
      auto val = ptup.at(i) + offset1;
      val += (val < cutoff) ? 0 : offset2;
      
      auto idx = i + offset1;
      idx += (idx < cutoff) ? 0 : offset2;

      ptup_full.at(idx) = val;
    }

    _permute_internal(this, ptup_full);

    qtnh::tidx_tup new_dims(totSize());
    for (std::size_t i = 0; i < totSize(); ++i) {
      new_dims.at(ptup_full.at(i)) = totDims().at(i);
    }

    auto [new_dis_dims, new_loc_dims] = utils::split_dims(new_dims, disDims().size());
    dis_dims_ = new_dis_dims;
    loc_dims_ = new_loc_dims;

    Broadcaster new_bc(bc().env, disSize(), { bc().str, bc().cyc, bc().off });
    bc_ = std::move(new_bc);

    return this;
  }

  SwapTensor::SwapTensor(const QTNHEnv& env, std::size_t n, std::size_t d)
    : SymmTensorBase(env, qtnh::tidx_tup(2 * d, n), qtnh::tidx_tup(4 - 2 * d, n), d) {}

  SwapTensor::SwapTensor(const QTNHEnv& env, std::size_t n, std::size_t d, BcParams params)
    : SymmTensorBase(env, qtnh::tidx_tup(2 * d, n), qtnh::tidx_tup(4 - 2 * d, n), d, params) {}

  std::unique_ptr<Tensor> SwapTensor::copy() const noexcept {
    auto n = dis_dims_.size() > 0 ? dis_dims_.at(0) : loc_dims_.at(0);
    auto tp = new SwapTensor(bc_.env, n, n_dis_in_dims_, { bc_.str, bc_.cyc, bc_.off });
    return std::unique_ptr<SwapTensor>(tp);
  }

  qtnh::tel SwapTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto dis_idxs = utils::i_to_idxs(bc_.group_id, dis_dims_);
    auto tot_idxs = utils::concat_dims(dis_idxs, loc_idxs);

    // CASE d = 1: 
    // 0 - | X | - 1
    // 2 - | X | - 3
    
    // CASE d = 0 or d = 2: 
    // 0 - | X | - 2
    // 1 - | X | - 3

    return (tot_idxs.at(0) == tot_idxs.at(3)) && (tot_idxs.at(1) == tot_idxs.at(2));
  }

  qtnh::tel SwapTensor::operator[](std::size_t i) const {
    auto loc_idxs = utils::i_to_idxs(i, loc_dims_);
    auto dis_idxs = utils::i_to_idxs(bc_.group_id, dis_dims_);
    auto tot_idxs = utils::concat_dims(dis_idxs, loc_idxs);

    // CASE d = 1: 
    // 0 - | X | - 1
    // 2 - | X | - 3
    
    // CASE d = 0 or d = 2: 
    // 0 - | X | - 2
    // 1 - | X | - 3

    return (tot_idxs.at(0) == tot_idxs.at(3)) && (tot_idxs.at(1) == tot_idxs.at(2));
  }

  qtnh::tel SwapTensor::at(qtnh::tidx_tup tot_idxs) const {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    if (bc_.group_id != (int)utils::idxs_to_i(dis_idxs, dis_dims_)) {
      throw std::invalid_argument("Element at given indices is not present on calling rank. ");
    }

    return (tot_idxs.at(0) == tot_idxs.at(3)) && (tot_idxs.at(1) == tot_idxs.at(2));
  }

  SwapTensor* SwapTensor::rebcast(BcParams params) {
    Broadcaster new_bc(bc_.env, bc_.base, params);
    bc_ = std::move(new_bc);

    return this;
  }
}
