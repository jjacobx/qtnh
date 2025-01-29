#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>

#include "tensor/tensor.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  Tensor::Tensor(const QTNHEnv& env) 
    : Tensor(env, qtnh::tidx_tup(), qtnh::tidx_tup()) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims)
    : Tensor(env, dis_dims, loc_dims, BcParams { 1, 1, 0 }) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, BcParams params)
    : dis_dims_(dis_dims), loc_dims_(loc_dims), bc_(env, qtnh::uint(utils::dims_to_size(dis_dims)), params, true) {}

  template<> 
  bool Tensor::canConvert<DenseTensor>() {
    return isDense(); 
  }
  template<> 
  bool Tensor::canConvert<SymmTensor>() {
    return isSymm(); 
  }
  template<> 
  bool Tensor::canConvert<DiagTensor>() {
    return isDiag(); 
  }

  bool Tensor::has(qtnh::tidx_tup tot_idxs) const {
    if (!bc_.isActive()) return false;

    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());

    (void)loc_idxs; // unused
    return (int)utils::idxs_to_i(dis_idxs, dis_dims_) == bc_.gid();
  }

  qtnh::tel Tensor::fetch(qtnh::tidx_tup tot_idxs) const {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());

    auto i = utils::idxs_to_i(dis_idxs, dis_dims_);
    auto r = i * bc_.params().str + bc_.params().off;

    qtnh::tel el;
    if (bc_.env().proc_id == r)
      el = (*this)[loc_idxs];
    
    MPI_Bcast(&el, 1, MPI_C_DOUBLE_COMPLEX, int(r), MPI_COMM_WORLD);

    return el;
  }

  Broadcaster::Broadcaster(Broadcaster&& bc)
    : Broadcaster(bc.env_, bc.base_, bc.params(), false) {
    is_active_ = bc.is_active_;
    if (bc.has_comm_) {
      gcomm_ = MPI_COMM_NULL;
      std::swap(gcomm_, bc.gcomm_);
      has_comm_ = true;
    }
  }

  Broadcaster::Broadcaster(const QTNHEnv &env, qtnh::uint base, BcParams params)
    : Broadcaster(env, base, params, true) {}

  Broadcaster::Broadcaster(const QTNHEnv &env, qtnh::uint base, BcParams params, bool communicate) 
    : env_(env), base_(base), str_(params.str), cyc_(params.cyc), off_(params.off) {
    int rel_id = env.proc_id - off_; // ! relative ID may be negative
    is_active_ = (rel_id >= 0) && (rel_id < (int)(str_ * cyc_ * base_));

    if (communicate) createComm();
  }

  Broadcaster& Broadcaster::operator=(Broadcaster&& b) noexcept {
    base_ = b.base_;
    str_ = b.str_;
    cyc_ = b.cyc_;
    off_ = b.off_;

    if (gcomm_ != MPI_COMM_NULL)
      MPI_Comm_free(&gcomm_);

    gcomm_ = MPI_COMM_NULL;
    std::swap(gcomm_, b.gcomm_);

    gid_ = b.gid_;
    is_active_ = b.is_active_;
    has_comm_ = b.has_comm_;

    return *this;
  }

  
  // In case there is a limited communicator pool, they should be actively freed
  Broadcaster::~Broadcaster() {
    deleteComm();
  }

  void Broadcaster::createComm() {
    #ifdef DEBUG
      if (utils::is_root()) std::cout << "CREATING GROUP_COMM\n";
    #endif

    MPI_Group world_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);

    // Create a group with active ranks. 
    std::vector<int> active_ids(str_ * cyc_ * base_);
    std::iota(active_ids.begin(), active_ids.end(), off_);
    MPI_Group active_group;
    MPI_Group_incl(world_group, int(active_ids.size()), active_ids.data(), &active_group);

    // Group communicator can be set up only on active ranks. 
    if (is_active_) {
      MPI_Comm active_comm;
      MPI_Comm_create_group(MPI_COMM_WORLD, active_group, 0, &active_comm);

      int rel_id = env_.proc_id - off_;
      int colour = (rel_id / (base_ * str_)) * str_ + rel_id % str_;
      MPI_Comm_split(active_comm, colour, rel_id, &gcomm_);
      MPI_Comm_rank(gcomm_, &gid_);

      MPI_Comm_free(&active_comm);
    }

    MPI_Group_free(&world_group);
    MPI_Group_free(&active_group);

    ++QTNHEnv::num_comms;
    has_comm_ = true;
  }

  void Broadcaster::deleteComm() {
    #ifdef DEBUG
      if (utils::is_root()) std::cout << "FREEING GROUP_COMM\n";
    #endif
    
    if (gcomm_ != MPI_COMM_NULL) MPI_Comm_free(&gcomm_);

    if (has_comm_) --QTNHEnv::num_comms;
    has_comm_ = false;
  }

  namespace ops {
    std::ostream& operator<<(std::ostream& out, const Tensor& o) {
      if (!o.bc().isActive()) {
        out << "Inactive";
        return out;
      }

      out << std::setprecision(2);

      TIndexing ti(o.totDims());
      for (auto idxs : ti.tup()) {
        if (o.has(idxs)) {
          out << o.at(idxs);
          if ((utils::idxs_to_i(idxs, o.totDims()) + 1) % o.locSize() != 0) {
            out << ", ";
          }
        }
      }

      return out;
    }
  }
}
