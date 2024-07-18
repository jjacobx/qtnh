#include <iomanip>
#include <mpi.h>

#include "tensor-new/tensor.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  Tensor::Tensor(const QTNHEnv& env) 
    : Tensor(env, qtnh::tidx_tup(), qtnh::tidx_tup()) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims)
    : Tensor(env, loc_dims, dis_dims, BcParams { 1, 1, 0 }) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, BcParams params)
    : loc_dims_(loc_dims), dis_dims_(dis_dims), bc_(env, utils::dims_to_size(dis_dims), params) {}

  bool Tensor::has(qtnh::tidx_tup tot_idxs) const {
    if (!bc_.active) return false;

    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());

    (void)loc_idxs; // unused
    return (int)utils::idxs_to_i(dis_idxs, dis_dims_) == bc_.group_id;
  }

  qtnh::tel Tensor::fetch(qtnh::tidx_tup tot_idxs) const {
    // TODO: MPI_Send of nearest element
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    return (*this)[loc_idxs]; // Temporary – return local element
  }

  Tensor::Broadcaster::Broadcaster(const QTNHEnv &env, qtnh::uint base, BcParams params) 
    : env(env), base(base), str(params.str), cyc(params.cyc), off(params.off) {
      int rel_id = env.proc_id - off; // ! relative ID may be negative
      active = (rel_id >= 0) && (rel_id < (int)(str * cyc * base));
      
      // Group active ranks into a single communicator
      MPI_Comm active_comm;
      MPI_Comm_split(MPI_COMM_WORLD, active, rel_id, &active_comm);

      // Group communicator will be uninitialised on inactive ranks
      if (active) {
        int colour = (rel_id / (base * str)) * str + rel_id % str;
        MPI_Comm_split(active_comm, colour, rel_id, &group_comm);
        MPI_Comm_rank(group_comm, &group_id);
      }

      MPI_Comm_free(&active_comm);
    }

  Tensor::Broadcaster& Tensor::Broadcaster::operator=(Broadcaster&& b) noexcept {
    base = b.base;
    str = b.str;
    cyc = b.cyc;
    off = b.off;

    group_comm = MPI_COMM_NULL;
    std::swap(group_comm, b.group_comm);

    group_id = b.group_id;
    active = b.active;

    return *this;
  }
  
  // In case there is a limited communicator pool, they should be actively freed
  Tensor::Broadcaster::~Broadcaster() {
    MPI_Comm_free(&group_comm);
  }

  namespace ops {
    std::ostream& operator<<(std::ostream& out, const Tensor& o) {
      if (!o.bc().active) {
        out << "Inactive";
        return out;
      }

      out << std::setprecision(2);

      TIndexing ti(o.locDims());
      for (auto idxs : ti) {
        out << o[idxs];
        if (utils::idxs_to_i(idxs, o.locDims()) < o.locSize() - 1) {
          out << ", ";
        }
      }

      return out;
    }
  }
}
