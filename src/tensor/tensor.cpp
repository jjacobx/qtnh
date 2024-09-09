#include <iomanip>
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
    : dis_dims_(dis_dims), loc_dims_(loc_dims), bc_(env, utils::dims_to_size(dis_dims), params) {}

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
    if (!bc_.active) return false;

    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());

    (void)loc_idxs; // unused
    return (int)utils::idxs_to_i(dis_idxs, dis_dims_) == bc_.group_id;
  }

  qtnh::tel Tensor::fetch(qtnh::tidx_tup tot_idxs) const {
    auto [dis_idxs, loc_idxs] = utils::split_dims(tot_idxs, dis_dims_.size());

    auto i = utils::idxs_to_i(dis_idxs, dis_dims_);
    auto r = i * bc_.str + bc_.off;

    qtnh::tel el;
    if (bc_.env.proc_id == r)
      el = (*this)[loc_idxs];
    
    MPI_Bcast(&el, 1, MPI_C_DOUBLE_COMPLEX, r, MPI_COMM_WORLD);

    return el;
  }

  Tensor::Broadcaster::Broadcaster(const QTNHEnv &env, qtnh::uint base, BcParams params) 
    : env(env), base(base), str(params.str), cyc(params.cyc), off(params.off) {
      int rel_id = env.proc_id - off; // ! relative ID may be negative
      active = (rel_id >= 0) && (rel_id < (int)(str * cyc * base));

      MPI_Group world_group;
      MPI_Comm_group(MPI_COMM_WORLD, &world_group);

      // Create a group with active ranks. 
      std::vector<int> active_ids(str * cyc * base);
      std::iota(active_ids.begin(), active_ids.end(), off);
      MPI_Group active_group;
      MPI_Group_incl(world_group, active_ids.size(), active_ids.data(), &active_group);

      // Group communicator can be set up only on active ranks. 
      if (active) {
        MPI_Comm active_comm;
        MPI_Comm_create_group(MPI_COMM_WORLD, active_group, 0, &active_comm);

        int colour = (rel_id / (base * str)) * str + rel_id % str;
        MPI_Comm_split(active_comm, colour, rel_id, &group_comm);
        MPI_Comm_rank(group_comm, &group_id);

        MPI_Comm_free(&active_comm);
      }
    }

  Tensor::Broadcaster& Tensor::Broadcaster::operator=(Broadcaster&& b) noexcept {
    base = b.base;
    str = b.str;
    cyc = b.cyc;
    off = b.off;

    group_comm = MPI_COMM_NULL;
    std::swap(group_comm, b.group_comm);

    group_id = b.group_id;
    active = false;
    std::swap(active, b.active);

    return *this;
  }
  
  // In case there is a limited communicator pool, they should be actively freed
  Tensor::Broadcaster::~Broadcaster() {
    if (active) MPI_Comm_free(&group_comm);
  }

  namespace ops {
    std::ostream& operator<<(std::ostream& out, const Tensor& o) {
      if (!o.bc().active) {
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
