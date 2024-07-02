#include <iomanip>
#include <mpi.h>

#include "core/utils.hpp"
#include "tensor/base2.hpp"
#include "tensor/indexing.hpp"

namespace qtnh {
  Tensor::Tensor(const QTNHEnv& env) 
    : Tensor(env, qtnh::tidx_tup(), qtnh::tidx_tup()) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims)
    : Tensor(env, loc_dims, dis_dims, 1, 1, 0) {}

  Tensor::Tensor(const QTNHEnv& env, qtnh::tidx_tup loc_dims, qtnh::tidx_tup dis_dims, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset)
    : dist_(env, stretch, cycles, offset, utils::dims_to_size(dis_dims)), loc_dims_(loc_dims), dis_dims_(dis_dims) {}


  qtnh::tel Tensor::fetch(const qtnh::tidx_tup& tot_idxs) const {
    // TODO: MPI_Send of nearest element
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());
    return (*this)[loc_idxs]; // Temporary – return local element
  }

  qtnh::Tensor::Distributor::Distributor(const QTNHEnv &env, qtnh::uint stretch, qtnh::uint cycles, qtnh::uint offset, qtnh::uint base) 
    : env(env), stretch(stretch), cycles(cycles), offset(offset), base(base) {
      int rel_id = env.proc_id - offset; // ! relative ID may be negative
      active = (rel_id >= 0) && (rel_id < stretch * cycles * base);
      
      // Group active ranks into a single communicator
      MPI_Comm active_comm;
      MPI_Comm_split(MPI_COMM_WORLD, active, rel_id, &active_comm);

      // Group communicator will be uninitialised on inactive ranks
      if (active) {
        int colour = rel_id / base + rel_id % stretch;
        MPI_Comm_split(active_comm, colour, rel_id, &group_comm);
      }
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
