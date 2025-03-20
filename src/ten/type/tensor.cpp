#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <numeric>

#include "ten/type/dense.hpp"
#include "ten/type/tensor.hpp"
#include "util/indexing.hpp"
#include "util/ops.hpp"

#ifndef AUTO_COMM_INIT
#define AUTO_COMM_INIT 0
#endif

namespace qtnh {
  Tensor::Tensor(const QTNHEnv& env)
  : Tensor(env, qtnh::tidx_tup(), qtnh::tidx_tup()) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims)
  : Tensor(env, dis_dims, loc_dims, BcParams { 1, 1, 0 }) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims, 
                 BcParams params)
  : dis_dims_(dis_dims)
  , loc_dims_(loc_dims)
  , bc_(env, qtnh::uint(utils::dims_to_size(dis_dims)), params, AUTO_COMM_INIT)
  {}

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

  void Tensor::reshape(qtnh::tidx_tup dis_dims, qtnh::tidx_tup loc_dims) {
    if (utils::compatible(dis_dims, dis_dims_) && utils::compatible(loc_dims, loc_dims_)) {
      dis_dims_ = dis_dims;
      loc_dims_ = loc_dims;
    } else {
      throw std::invalid_argument("Incompatible new dimensions.");
    }
  }

  void Tensor::print_serial(std::string name, bool skip_inactive) {
    auto tp = Tensor::convert<DenseTensor>(this->copy());
    auto loc_els = tp->extractEls();

    MPI_Request req1 = MPI_REQUEST_NULL;
    MPI_Request req2 = MPI_REQUEST_NULL;
    auto is_active = bc_.isActive();
    MPI_Isend(&is_active, 1, MPI_CXX_BOOL, 0, 0, MPI_COMM_WORLD, &req1);
    
    if (is_active) {
      MPI_Isend(loc_els.data(), int(locSize()), MPI_DOUBLE_COMPLEX, 0, 0, 
                MPI_COMM_WORLD, &req2);
    }

    if (utils::is_root()) {
      for (auto i = 0UL; i < bc_.env().num_processes; ++i) {
        bool is_active_target;
        MPI_Recv(&is_active_target, 1, MPI_CXX_BOOL, int(i), 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        if (is_active_target) {
          std::vector<tel> loc_els_target(locSize());
          MPI_Recv(loc_els_target.data(), int(locSize()), MPI_DOUBLE_COMPLEX, int(i), 0, 
                   MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          
          std::cout << "P" << i << " | " << name << " = " << loc_els_target << "\n";
        } else if (!skip_inactive) {
          std::cout << "P" << i << " | " << name << " = Inactive\n";
        }
      }
    }

    MPI_Wait(&req1, MPI_STATUS_IGNORE);
    MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if (req1 != MPI_REQUEST_NULL) MPI_Request_free(&req1);
    if (req2 != MPI_REQUEST_NULL) MPI_Request_free(&req2);
  }


  Broadcaster::Broadcaster(Broadcaster &&b)
  : Broadcaster(b.env_, b.base_, b.params(), false)
  {
    std::swap(gcomm_, b.gcomm_);
    std::swap(has_comm_, b.has_comm_);
  }

  Broadcaster::Broadcaster(const QTNHEnv &env, qtnh::uint base, BcParams params)
  : Broadcaster(env, base, params, true)
  {}

  Broadcaster::Broadcaster(const QTNHEnv &env, qtnh::uint base, BcParams params, bool communicate)
  : env_(env)
  , base_(base)
  , str_(params.str)
  , cyc_(params.cyc)
  , off_(params.off) 
  {
    int rel_id = env.proc_id - off_; // ! relative ID may be negative
    is_active_ = (rel_id >= 0) && (rel_id < (int)(str_ * cyc_ * base_));
    if (is_active_) gid_ = (rel_id / str_) % base_;
    if (communicate) createComm();
    // if (env.proc_id == 0) std::cout << "Communicate: " << communicate << "\n";
  }

  Broadcaster& Broadcaster::operator=(Broadcaster&& b) noexcept {
    std::swap(base_, b.base_);
    std::swap(str_, b.str_);
    std::swap(cyc_, b.cyc_);
    std::swap(off_, b.off_);

    std::swap(gid_, b.gid_);
    std::swap(is_active_, b.is_active_);

    // ! The moved broadcaster should be deleted. 
    // ! Swapping communicators will hopefully free the old one. 
    std::swap(gcomm_, b.gcomm_);
    std::swap(has_comm_, b.has_comm_);

    return *this;
  }

  // In case there is a limited communicator pool, they should be actively freed
  Broadcaster::~Broadcaster() {
    deleteComm();
  }

  const MPI_Comm& Broadcaster::gcomm() {
    if (!has_comm_) createComm();
    return gcomm_;
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

      // TODO: Remove if everything works well. 
      int test_gid;
      MPI_Comm_rank(gcomm_, &test_gid);
      if (gid_ != test_gid) {
        std::cout << "gid = " << gid_ << " but should be " << test_gid << "\n";
        MPI_Abort(gcomm_, 25);
      }

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
}
