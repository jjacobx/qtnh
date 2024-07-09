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
    if (offset == 0) {
      return this;
    } else if (offset < 0) {
      if (!bc_.active) return this;
      
      auto loc_dims2 = qtnh::tidx_tup(dis_dims_.end() + offset, dis_dims_.end());
      auto shift = utils::dims_to_size(loc_dims2);

      MPI_Comm gath_comm;
      MPI_Comm_split(bc_.group_comm, bc_.group_id / shift, bc_.group_id, &gath_comm);

      loc_els_.resize(locSize() * shift);
      MPI_Allgather(loc_els_.data(), locSize(), MPI_C_DOUBLE_COMPLEX, 
                    loc_els_.data(), locSize(), MPI_C_DOUBLE_COMPLEX, gath_comm);
      
      dis_dims_.erase(dis_dims_.end() + offset, dis_dims_.end());
      loc_dims_.insert(loc_dims_.begin(), loc_dims2.begin(), loc_dims2.end());

      Broadcaster new_bc(bc_.env, locSize(), { bc_.str * shift, bc_.cyc, bc_.off });
      bc_ = std::move(new_bc);

      return this;
    } else {
      auto dis_dims2 = qtnh::tidx_tup(loc_dims_.begin(), loc_dims_.begin() + offset);
      auto shift = utils::dims_to_size(dis_dims2);

      // Align with multiples of front – should happen in place
      rebcast({ std::max(shift, (bc_.str / shift) * shift), bc_.cyc, bc_.off });

      if (!bc_.active) return this;

      auto split_id = (((bc_.env.proc_id - bc_.off) % (bc_.base * bc_.str)) / (bc_.str / shift)) % shift;
      loc_els_.erase(loc_els_.begin(), loc_els_.begin() + locSize() / shift * split_id);
      loc_els_.erase(loc_els_.begin() + locSize() / shift, loc_els_.end());

      loc_dims_.erase(loc_dims_.begin(), loc_dims_.begin() + offset);
      dis_dims_.insert(dis_dims_.end(), dis_dims2.begin(), dis_dims2.end());

      Broadcaster new_bc(bc_.env, locSize(), { bc_.str / shift, bc_.cyc, bc_.off });
      bc_ = std::move(new_bc);

      return this;
    }
  }
}
