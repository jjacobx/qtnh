#include <iomanip>
#include <mpi.h>

#include "tensor-new/tensor.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  Tensor::Tensor(const QTNHEnv& env) 
    : Tensor(env, qtnh::tidx_tup(), qtnh::tidx_tup()) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims)
    : Tensor(env, loc_dims, dis_dims, DistParams { 1, 1, 0 }) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, DistParams params)
    : dist_(env, utils::dims_to_size(dis_dims), params), loc_dims_(loc_dims), dis_dims_(dis_dims) {}


  qtnh::tel Tensor::fetch(qtnh::tidx_tup tot_idxs) const {
    // TODO: MPI_Send of nearest element
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    return (*this)[loc_idxs]; // Temporary – return local element
  }

  Tensor::Distributor::Distributor(const QTNHEnv &env, qtnh::uint base, DistParams params) 
    : env(env), base(base), stretch(params.stretch), cycles(params.cycles), offset(params.offset) {
      int rel_id = env.proc_id - offset; // ! relative ID may be negative
      active = (rel_id >= 0) && (rel_id < stretch * cycles * base);
      
      // Group active ranks into a single communicator
      MPI_Comm active_comm;
      MPI_Comm_split(MPI_COMM_WORLD, active, rel_id, &active_comm);

      // Group communicator will be uninitialised on inactive ranks
      if (active) {
        int colour = (rel_id / (base * stretch)) * stretch + rel_id % stretch;
        MPI_Comm_split(active_comm, colour, rel_id, &group_comm);
        MPI_Comm_rank(group_comm, &group_id);
      }

      MPI_Comm_free(&active_comm);
    }
  
  // In case there is a limited communicator pool, they should be actively freed
  Tensor::Distributor::~Distributor() {
    MPI_Comm_free(&group_comm);
  }

  namespace ops {
    std::ostream& operator<<(std::ostream& out, const Tensor& o) {
      if (!o.dist().active) {
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
