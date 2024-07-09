#include "tensor-new/dense.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  DenseTensorBase::DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims)
    : Tensor(env, loc_dims, dis_dims) {}

  DenseTensorBase::DenseTensorBase(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, BcParams params)
    : Tensor(env, loc_dims, dis_dims, params) {}

  DenseTensor* DenseTensorBase::toDense() {
    std::vector<qtnh::tel> els;
    els.reserve(locSize());

    TIndexing ti(locDims());
    for (auto idxs : ti) {
      els.push_back((*this)[idxs]);
    }

    // ? Is it better to use local members or accessors? 
    return new DenseTensor(bc_.env, loc_dims_, dis_dims_, std::move(els), BcParams { bc_.str, bc_.cyc, bc_.off });
  }

  DenseTensor::DenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel>&& els)
    : DenseTensorBase(env, loc_dims, dis_dims), TIDense(std::move(els)) {}

  DenseTensor::DenseTensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, std::vector<qtnh::tel>&& els, BcParams params)
    : DenseTensorBase(env, loc_dims, dis_dims, params), TIDense(std::move(els)) {}

  qtnh::tel DenseTensor::operator[](qtnh::tidx_tup loc_idxs) const {
    auto i = utils::idxs_to_i(loc_idxs, loc_dims_);
    return loc_els_.at(i);
  }

  qtnh::tel& DenseTensor::operator[](qtnh::tidx_tup loc_idxs) {
    auto i = utils::idxs_to_i(loc_idxs, loc_dims_);
    return loc_els_.at(i);
  }

  void DenseTensor::put(qtnh::tidx_tup tot_idxs, qtnh::tel el) {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    auto target_id = utils::idxs_to_i(dis_idxs, dis_dims_);

    int call_id;
    MPI_Comm_rank(bc_.group_comm, &call_id);

    if (call_id == target_id) {
      auto i = utils::idxs_to_i(loc_idxs, loc_dims_);
      loc_els_.at(i) = el;
    }
  }

  DenseTensor* DenseTensor::swap(qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    _swap_internal(this, idx1, idx2);
    return this;
  }

  DenseTensor* DenseTensor::rebcast(BcParams params) {
    _rebcast_internal(this, params);
    
    // Update broadcaster
    Broadcaster new_bc(bc_.env, bc_.base, params);
    bc_ = std::move(new_bc);

    return this;
  }

  DenseTensor* DenseTensor::rescatter(int offset) {
    _rescatter_internal(this, offset);

    // Update dimensions and broadcaster
    if (offset < 0) {
      auto loc_dims2 = qtnh::tidx_tup(dis_dims_.end() + offset, dis_dims_.end());
      auto shift = utils::dims_to_size(loc_dims2);

      loc_dims_.insert(loc_dims_.begin(), loc_dims2.begin(), loc_dims2.end());
      dis_dims_.erase(dis_dims_.end() - offset, dis_dims_.end());

      Broadcaster new_bc(bc_.env, locSize(), { bc_.str * shift, bc_.cyc, bc_.off });
      bc_ = std::move(new_bc);
    } else if (offset > 0) {
      auto dis_dims2 = qtnh::tidx_tup(loc_dims_.begin(), loc_dims_.begin() + offset);
      auto shift = utils::dims_to_size(dis_dims2);

      BcParams params(std::max(1UL, bc_.str / shift), bc_.cyc, bc_.off);

      loc_dims_.erase(loc_dims_.begin(), loc_dims_.begin() + offset);
      dis_dims_.insert(dis_dims_.end(), dis_dims2.begin(), dis_dims2.end());

      Broadcaster new_bc(bc_.env, locSize(), params);
      bc_ = std::move(new_bc);
    }

    return this;
  }

  void TIDense::_swap_internal(Tensor* target, qtnh::tidx_tup_st idx1, qtnh::tidx_tup_st idx2) {
    auto& bc = target->bc();

    if (!bc.active) return;
    if (idx1 > idx2) std::swap(idx1, idx2);

    // Case: asymmetric swap
    if (target->totDims().at(idx1) != target->totDims().at(idx2)) {
      throw std::runtime_error("Asymmetric swaps are currently not allowed");
    }

    #ifdef DEBUG
      std::cout << "Swapping " << idx1 << " and " << idx2 << "\n";
    #endif

    // Case: same-index swap
    if (idx1 == idx2) return;

    // Case: local swap
    if (idx1 >= target->disDims().size()) {
      qtnh::tidx_tup_st loc_idx1 = idx1 - target->disDims().size();
      qtnh::tidx_tup_st loc_idx2 = idx2 - target->disDims().size();

      auto loc_dims = target->locDims();
      qtnh::tifl_tup ifls(loc_dims.size(), { TIdxT::open, 0 });
      ifls.at(loc_idx1) = ifls.at(loc_idx2) = { TIdxT::closed, 0 };
      TIndexing ti(loc_dims, ifls);

      for (auto idxs : ti) {
        auto idxs1 = idxs;
        auto idxs2 = idxs;

        // Swaps only invoked n * (n - 1) / 2 times, instead of n * n
        for (qtnh::tidx i = 0; i < loc_dims.at(loc_idx1) - 1; ++i) {
          idxs1.at(loc_idx1) = idxs2.at(loc_idx2) = i;
          for (qtnh::tidx j = i + 1; j < loc_dims.at(loc_idx2); ++j) {
            idxs1.at(loc_idx2) = idxs2.at(loc_idx1) = j;

            auto i1 = utils::idxs_to_i(idxs1, loc_dims);
            auto i2 = utils::idxs_to_i(idxs2, loc_dims);

            std::swap(loc_els_.at(i1), loc_els_.at(i2));
          }
        }
      }

      return;
    }

    // Case: mixed local/distributed swap
    if (idx1 < target->disDims().size() && idx2 >= target->disDims().size()) {
      auto dims = target->totDims();
      qtnh::tidx_tup trail_dims(dims.begin() + idx2 + 1, dims.end());
      auto block_length = utils::dims_to_size(trail_dims);
      auto stride = dims.at(idx2) * block_length;

      qtnh::tidx_tup mid_loc_dims(dims.begin() + target->disDims().size(), dims.begin() + idx2);
      auto num_blocks = utils::dims_to_size(mid_loc_dims);

      qtnh::tidx_tup mid_dist_dims(target->disDims().begin() + idx1 + 1, target->disDims().end());
      auto dist_stride = utils::dims_to_size(mid_dist_dims);

      auto dist_idxs = utils::i_to_idxs(target->bc().group_id, target->disDims());
      auto rank_idx = dist_idxs.at(idx1);

      MPI_Datatype strided, restrided;
      MPI_Type_vector(num_blocks, block_length, stride, MPI_C_DOUBLE_COMPLEX, &strided);
      MPI_Type_create_resized(strided, 0, block_length * sizeof(qtnh::tel), &restrided);
      MPI_Type_commit(&restrided);

      MPI_Comm swap_comm;
      MPI_Comm_split(bc.group_comm, bc.group_id - rank_idx * dist_stride, bc.group_id, &swap_comm);

      std::vector<qtnh::tel> new_els(loc_els_.size());
      for (std::size_t i = 0; i < dims.at(idx1); ++i) {
        // TODO: Consider MPI message size limit. 
        // * A scatter might already take it into account
        MPI_Scatter(loc_els_.data(), 1, restrided, new_els.data() + i * block_length, 1, restrided, i, swap_comm);
      }

      // ! new_els should not be copied, and original loc_els should be destroyed. 
      loc_els_ = std::move(new_els);

      MPI_Comm_free(&swap_comm);
      MPI_Type_free(&restrided);
      return;
    }

    // Case: distributed swap
    if (idx2 < target->disDims().size()) {
      auto target_idxs = utils::i_to_idxs(bc.group_id, target->disDims());
      std::swap(target_idxs.at(idx1), target_idxs.at(idx2));
      auto target_i = utils::idxs_to_i(target_idxs, target->disDims());

      std::vector<qtnh::tel> new_els(loc_els_.size());
      // TODO: Consider MPI message size limit – not a scatter. 
      MPI_Sendrecv(loc_els_.data(), loc_els_.size(), MPI_C_DOUBLE_COMPLEX, target_i, 0, 
                   new_els.data(), new_els.size(), MPI_C_DOUBLE_COMPLEX, target_i, 0, bc.group_comm, MPI_STATUS_IGNORE);
      
      // ! new_els should not be copied, and original loc_els should be destroyed
      loc_els_ = std::move(new_els);
      return;
    }
  }

  void TIDense::_rebcast_internal(Tensor* target, BcParams params) {
    auto& bc = target->bc();
    Tensor::Broadcaster new_bc(bc.env, bc.base, params);
    std::vector<MPI_Request> send_reqs(params.str * params.cyc);

    if (bc.active) {
      std::vector<int> send_sources;
      std::vector<int> send_targets;

      for (int i = 0; i < bc.str; ++i) {
        for (int j = 0; j < bc.cyc; ++j) {
          send_sources.push_back(i + (bc.base * j + bc.group_id) * bc.str + bc.off);
        }
      }

      for (int i = 0; i < params.str; ++i) {
        for (int j = 0; j < params.cyc; ++j) {
          send_targets.push_back(i + (bc.base * j + bc.group_id) * params.str + params.off);
        }
      }

      // TODO: optimisation where if data is already present at target, it is not sent. 
      if (bc.env.proc_id == send_sources.at(0)) {
        for (int i = 0; i < send_targets.size(); ++i) {
          MPI_Isend(loc_els_.data(), loc_els_.size(), MPI_C_DOUBLE_COMPLEX, send_targets.at(i), 0, MPI_COMM_WORLD, &send_reqs.at(i));
        }
      }
    }

    std::vector<qtnh::tel> new_els(utils::dims_to_size(target->locDims()));
    if (new_bc.active) {
      int recv_source = new_bc.group_id * bc.str + bc.off;
      MPI_Recv(new_els.data(), new_els.size(), MPI_C_DOUBLE_COMPLEX, recv_source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      loc_els_ = std::move(new_els);
    }

    MPI_Waitall(send_reqs.size(), send_reqs.data(), MPI_STATUSES_IGNORE);
    if (!new_bc.active) loc_els_.clear();

    return;
  }

  void TIDense::_rescatter_internal(Tensor* target, int offset) {
    auto& bc = target->bc();

   if (offset < 0) {
      if (!bc.active) return;
      
      auto loc_dims2 = qtnh::tidx_tup(target->disDims().end() + offset, target->disDims().end());
      auto shift = utils::dims_to_size(loc_dims2);

      MPI_Comm gath_comm;
      MPI_Comm_split(bc.group_comm, bc.group_id / shift, bc.group_id, &gath_comm);

      loc_els_.resize(target->locSize() * shift);
      MPI_Allgather(loc_els_.data(), target->locSize(), MPI_C_DOUBLE_COMPLEX, 
                    loc_els_.data(), target->locSize(), MPI_C_DOUBLE_COMPLEX, gath_comm);
    } else if (offset > 0) {
      auto dis_dims2 = qtnh::tidx_tup(target->locDims().begin(), target->locDims().begin() + offset);
      auto shift = utils::dims_to_size(dis_dims2);

      // Align with multiples of shift
      BcParams params(std::max(shift, (bc.str / shift) * shift), bc.cyc, bc.off);
      _rebcast_internal(target, params);

      Tensor::Broadcaster bc2(bc.env, bc.base, params);
      if (!bc2.active) return;

      auto split_id = (((bc2.env.proc_id - bc2.off) % (bc2.base * bc2.str)) / (bc2.str / shift)) % shift;
      loc_els_.erase(loc_els_.begin(), loc_els_.begin() + target->locSize() / shift * split_id);
      loc_els_.erase(loc_els_.begin() + target->locSize() / shift, loc_els_.end());
    }
  }
}