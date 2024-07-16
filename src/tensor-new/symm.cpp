#include "tensor-new/symm.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  SymmTensorBase::SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims) 
    : DenseTensorBase(env, loc_dims, dis_dims), n_dis_in_dims_(n_dis_in_dims) {}

  SymmTensorBase::SymmTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, BcParams params) 
    : DenseTensorBase(env, loc_dims, dis_dims, params), n_dis_in_dims_(n_dis_in_dims) {}

  SymmTensor* SymmTensorBase::toSymm() {
    std::vector<qtnh::tel> els;
    els.reserve(locSize());

    TIndexing ti(locDims());
    for (auto idxs : ti) {
      els.push_back((*this)[idxs]);
    }

    return new SymmTensor(bc_.env, loc_dims_, dis_dims_, n_dis_in_dims_, std::move(els), BcParams { bc_.str, bc_.cyc, bc_.off });
  }

  SymmTensor::SymmTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel>&& els) 
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims), TIDense(std::move(els)) {}

  SymmTensor::SymmTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::tidx_tup_st n_dis_in_dims, std::vector<qtnh::tel>&& els, BcParams params) 
    : SymmTensorBase(env, loc_dims, dis_dims, n_dis_in_dims, params), TIDense(std::move(els)) {}


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
    if (bc_.group_id != utils::idxs_to_i(dis_idxs, dis_dims_)) {
      throw std::invalid_argument("Element at given indices is not present on calling rank. ");
    }

    return loc_els_.at(utils::idxs_to_i(loc_idxs, loc_dims_));
  }

  qtnh::tel& SymmTensor::at(qtnh::tidx_tup tot_idxs) {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    if (bc_.group_id != utils::idxs_to_i(dis_idxs, dis_dims_)) {
      throw std::invalid_argument("Element at given indices is not present on calling rank. ");
    }

    return loc_els_.at(utils::idxs_to_i(loc_idxs, loc_dims_));
  }

  void SymmTensor::put(qtnh::tidx_tup tot_idxs, qtnh::tel el) {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    auto target_id = utils::idxs_to_i(dis_idxs, dis_dims_);

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

  SymmTensor* SymmTensor::rescatter(int offset, TIdxIO io) {
    if (offset < 0) {
      if (io == TIdxIO::in) {
        _shift_internal(this, disInDims().size() + offset, disDims().size(), offset);
        n_dis_in_dims_ += offset;
      }

      _rescatter_internal(this, offset);

      auto loc_dims2 = qtnh::tidx_tup(dis_dims_.end() + offset, dis_dims_.end());
      auto shift = utils::dims_to_size(loc_dims2);

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

  SwapTensor::SwapTensor(const QTNHEnv& env, std::size_t n, std::size_t d)
    : SymmTensorBase(env, qtnh::tidx_tup(2 * d, n), qtnh::tidx_tup(4 - 2 * d, n), d) {}

  SwapTensor::SwapTensor(const QTNHEnv& env, std::size_t n, std::size_t d, BcParams params)
    : SymmTensorBase(env, qtnh::tidx_tup(2 * d, n), qtnh::tidx_tup(4 - 2 * d, n), d, params) {}

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
    if (bc_.group_id != utils::idxs_to_i(dis_idxs, dis_dims_)) {
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
